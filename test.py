from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode

#SparkSession.builder.master("local[*]").getOrCreate().stop()
spark = SparkSession.builder.appName("LoRaWAN Anomaly Detection").master("local[*]").getOrCreate()


# Load JSON data
df = spark.read.json("./dataset_test/test1.log")

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

# Show the resulting DataFrame
df.show(truncate=False)

# Save the DataFrame to a CSV file
output_path = "./output_csv"
df.write.option("header", "true").csv(output_path)

print(f"Results saved to {output_path}")

spark.stop()