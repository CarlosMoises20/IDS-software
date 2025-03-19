from pyspark.sql import SparkSession
from message_classification import MessageClassification as mc
from dataset_type import DatasetType
from concurrent.futures import ThreadPoolExecutor
from constants import *


def execute(dataset_type, spark_session):
    
    # TODO: change return for a general confusion matrix??
    test_result = mc.message_classification(spark_session, dataset_type)

    output_path = f'./output_test_{dataset_type.value["name"]}'
    
    #test_result.write.mode("overwrite").csv(output_path)    # TODO: maybe parquet file instead ?? analyse it later

    return test_result


if __name__ == '__main__':

    spark_session = SparkSession.builder \
                            .appName(SPARK_APP_NAME) \
                            .config("spark.ui.port", SPARK_PORT) \
                            .config("spark.sql.shuffle.partitions", SPARK_NUM_PARTITIONS)  \
                            .config("spark.sql.files.maxPartitionBytes", SPARK_FILES_MAX_PARTITION_BYTES)  \
                            .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY) \
                            .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
                            .config("spark.executor.memoryOverhead", SPARK_EXECUTOR_MEMORY_OVERHEAD) \
                            .config("spark.network.timeout", SPARK_NETWORK_TIMEOUT) \
                            .config("spark.executor.heartbeatInterval", SPARK_EXECUTOR_HEARTBEAT_INTERVAL) \
                            .getOrCreate()
    
    spark_session.sparkContext.setLogLevel("DEBUG")

    # List of tasks (each task processes a different category of LoRaWAN messages according to the gateway)
    tasks = [DatasetType.RXPK, DatasetType.TXPK]

    # List to store the results
    results = []

    # Execute tasks in parallel using threads that use the Spark Session to process LoRaWAN data
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(execute, task, spark_session) for task in tasks]

        # Waits for conclusion of all tasks
        for future in futures:
            results.append(future.result())  


    # Stop the Spark Session
    spark_session.stop()