from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, explode
import os
from utils import *


# Dataset directory
dataset_directory = os.fsencode('.\dataset_test')

# Initialize Spark Session
spark = SparkSession.builder.appName("LoRaWAN Anomaly Detection").master("local[*]").getOrCreate()

# Output directory to store the results
output_path = "./output"

# object where all the summaries of the results will be stored
final_summary = None


# for each file inside the directory, process the messages 
# inside it according to the parameters on 'schema'
for file in os.listdir(dataset_directory):

    # absolute path
    filename = os.path.join(os.fsdecode(dataset_directory), os.fsdecode(file))
    
    # Load the data from the dataset file
    df = spark.read.json(filename)

    # Explode the 'rxpk' array
    df = df.withColumn("rxpk", explode(col("rxpk")))

    # Extract individual fields from 'rxpk', including 'rsig'
    for field in df.schema["rxpk"].dataType.fields:
        df = df.withColumn(field.name, col(f"rxpk.{field.name}"))

    # Drop the 'rxpk' column as it's now flattened
    df = df.drop("rxpk")

    # Explode the 'rsig' array
    df = df.withColumn("rsig", explode(col("rsig")))

    # Extract individual fields from 'rsig'
    for field in df.schema["rsig"].dataType.fields:
        df = df.withColumn(field.name, col(f"rsig.{field.name}"))

    # Drop the 'rsig' column as it's now flattened
    df = df.drop("rsig")

    # Add anomaly detection columns
    df = df.withColumn("Anomaly_SNR", when(col("lsnr") < -10, 1).otherwise(0)) \
        .withColumn("Anomaly_RSSI", when(col("rssi") < -120, 1).otherwise(0)) \
        .withColumn("Anomaly_MIC", when(col("MIC").isNull(), 1).otherwise(0)) \
        .withColumn("Anomaly_Size", when(col("size") < 10, 1).otherwise(0))

    # Combine anomaly indicators
    df = df.withColumn("Anomaly", (col("Anomaly_SNR") + col("Anomaly_RSSI") + col("Anomaly_MIC") + col("Anomaly_Size")) > 0)

    # Group data for analysis
    summary = df.groupBy("Anomaly").agg(
        count("*").alias("MessageCount"),
        count(when(col("Anomaly_SNR") == 1, 1)).alias("SNR_Anomalies"),
        count(when(col("Anomaly_RSSI") == 1, 1)).alias("RSSI_Anomalies"),
        count(when(col("Anomaly_MIC") == 1, 1)).alias("MIC_Anomalies"),
        count(when(col("Anomaly_Size") == 1, 1)).alias("Size_Anomalies")
    )

    # Concatenate summaries
    if final_summary is None:
        final_summary = summary
    else:
        final_summary = final_summary.union(summary)

    print(f"File '{filename}' has been processed")


final_summary.show(truncate=False)

# Save the final summary
final_summary.write.mode("overwrite").csv(output_path)

print(f"Anomaly summary saved to: {output_path}")

spark.stop()