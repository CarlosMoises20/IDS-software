
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import os


# Dataset directory
dataset_directory = os.fsencode('.\dataset_test')

# Initialize Spark Session
spark = SparkSession.builder.appName("LoRaWAN Anomaly Detection").master("local[*]").getOrCreate()

# Define the schema for the JSON messages
schema = StructType([
    StructField("tmst", IntegerType(), False),
    StructField("time", StringType(), False),
    StructField("chan", IntegerType(), False),
    StructField("rfch", IntegerType(), False),
    StructField("freq", DoubleType(), False),
    StructField("stat", IntegerType(), False),
    StructField("modu", StringType(), False),
    StructField("datr", StringType(), False),
    StructField("codr", StringType(), False),
    StructField("lsnr", DoubleType(), False),
    StructField("rssi", IntegerType(), False),
    StructField("size", IntegerType(), False),
    StructField("data", StringType(), False),
    StructField("MessageType", StringType(), False),
    StructField("PHYPayload", StringType(), False),
    StructField("MHDR", StringType(), False),
    StructField("MACPayload", StringType(), False),
    StructField("MIC", StringType(), False),
    StructField("FHDR", StringType(), False),
    StructField("FPort", StringType(), False),
    StructField("FRMPayload", StringType(), False),
    StructField("DevAddr", StringType(), False),
    StructField("FCtrl", StringType(), False),
    StructField("FCnt", StringType(), False),
    StructField("FOpts", StringType(), False),
    StructField("Direction", StringType(), False),
    StructField("FCtrlACK", StringType(), False),
    StructField("FCtrlADR", StringType(), False)
])

# Output directory to store the results
output_path = "./output"

# object where all the summaries of the results will be stored
final_summary = None

for file in os.listdir(dataset_directory):

    # absolute path
    filename = os.path.join(os.fsdecode(dataset_directory), os.fsdecode(file))
    
    # Load the data from the dataset path
    df = spark.read.schema(schema).json(filename)

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