
from pyspark.sql.functions import col, when, count, explode

### On this module, add functions, where each function process a different type of messages


# Converts a dataset of type 'rxpk', given the filename of the dataset, into a 'df' Spark dataframe
def pre_process_rxpk_dataset(spark_session, filename):

    # Load the data from the dataset file
    df = spark_session.read.json(filename)

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

    return df




# Converts a dataset of type 'txpk', given the filename of the dataset, into a 'df' Spark dataframe
def pre_process_txpk_dataset(spark_session, filename):

    ### PRE-PROCESSING

    # Load the data from the dataset file
    df = spark_session.read.json(filename)

    # Explode the 'txpk' array
    df = df.withColumn("txpk", explode(col("txpk")))

    # Extract individual fields from 'txpk'
    for field in df.schema["txpk"].dataType.fields:
        df = df.withColumn(field.name, col(f"txpk.{field.name}"))

    # Drop the 'txpk' column as it's now flattened
    df = df.drop("txpk")

    return df




def process_rxpk_dataset(spark_session, dataset_rxpk):

    #df = pre_process_rxpk_dataset(spark_session, filename)

    #return df
    pass




def process_txpk_dataset(spark_session, dataset_txpk):

    #df = pre_process_txpk_dataset(spark_session, filename) 

    #return df
    pass