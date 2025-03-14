from pyspark.sql import SparkSession
from message_classification import MessageClassification as mc
from dataset_type import DatasetType
from concurrent.futures import ThreadPoolExecutor
from crate.client import connect


def execute(dataset_type, spark_session):
    
    test_result = mc.message_classification(spark_session, dataset_type)

    output_path = f'./output_test_{dataset_type.value["filename_field"]}'
    
    #test_result.write.mode("overwrite").csv(output_path)    # TODO: maybe parquet file instead ?? analyse it later

    return test_result


if __name__ == '__main__':

    # Initialize Spark Session
    spark_session = SparkSession.builder \
                                .appName("IDS for LoRaWAN network") \
                                .config("spark.sql.shuffle.partitions", "400")  \
                                .config("spark.sql.autoBroadcastJoinThreshold", "-1")  \
                                .config("spark.sql.files.maxPartitionBytes", "134217728")  \
                                .config("spark.executor.memory", "8g") \
                                .config("spark.driver.memory", "8g") \
                                .config("spark.executor.memoryOverhead", "3000") \
                                .config("spark.network.timeout", "800s") \
                                .config("spark.executor.heartbeatInterval", "60s") \
                                .getOrCreate()

    # List of tasks (each task processes a different category of LoRaWAN messages according to the gateway)
    tasks = [DatasetType.RXPK, DatasetType.TXPK]

    # List to store the results
    results = []

    # Execute tasks in parallel using threads that use the Spark Session to process LoRaWAN data
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(execute, task, spark_session) for task in tasks]

        # Aguarda a conclusão de todas as tarefas
        for future in futures:
            print(future.result())
            results.append(future.result())  


    # TODO: store the two models & their results on CrateDB to be later used on real-time message processing; replace previous model by 
    # these ones


    # Stop the Spark Session
    spark_session.stop()