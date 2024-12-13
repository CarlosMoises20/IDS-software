
from pyspark.sql.functions import col, when, count, explode

### On this module, add functions, where each function process a different type of messages


# Process a dataset of type 'rxpk'
# TODO: correct the implementation (only anomaly part)
# process_type: indicates if the model is being trained or tested
def process_rxpk_dataset(df, process_type):
    
    # Explode the 'rxpk' array
    df = df.withColumn("rxpk", explode(col("rxpk")))

    # Extract individual fields from 'rxpk'
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

    ## TODO: correct this
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

    return summary


# Process a dataset of type 'stat'
# TODO: correct the implementation (only anomaly part)
# process_type: indicates if the model is being trained or tested
# and then considering each attack, call a specific function that will be in another module
def process_stat_dataset(df, process_type):

    # Explode the 'rxpk' array
    df = df.withColumn("rxpk", explode(col("rxpk")))

    # Extract individual fields from 'rxpk'
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


    # TODO: correct
    # Group data for analysis   
    
    # Add anomaly detection column
    #df = df.withColumn("Anomaly", when(col("count") < 10, 1).otherwise(0))

    #summary = df.groupBy("Anomaly").agg(
    #    count("*").alias("MessageCount"),
    #    count(when(col("Anomaly") == 1, 1)).alias("Anomalies")
    #)

    summary = 0

    return summary


# Process a dataset of type 'txpk'
# TODO: correct the implementation (only anomaly part)
# process_type: indicates if the model is being trained or tested
def process_txpk_dataset(df, process_type):

    # Explode the 'txpk' array
    df = df.withColumn("txpk", explode(col("txpk")))

    # Extract individual fields from 'txpk'
    for field in df.schema["txpk"].dataType.fields:
        df = df.withColumn(field.name, col(f"txpk.{field.name}"))

    # Drop the 'txpk' column as it's now flattened
    df = df.drop("txpk")

    # Add anomaly detection column
    # df = df.withColumn("Anomaly", when(col("count") < 10, 1).otherwise(0))

    # Group data for analysis
    #summary = df.groupBy("Anomaly").agg(
    #    count("*").alias("MessageCount"),
    #    count(when(col("Anomaly") == 1, 1)).alias("Anomalies")
    #)

    summary = 0

    return summary

