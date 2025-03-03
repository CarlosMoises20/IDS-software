from pyspark.sql import SparkSession
from message_classification import MessageClassification as mc
from dataset_type import DatasetType
from concurrent.futures import ThreadPoolExecutor


def execute(dataset_type, spark_session):
    
    test_result = mc.message_classification(spark_session, dataset_type)

    output_path = f'./output_test_{dataset_type.value["filename_field"]}'
    #test_result.write.mode("overwrite").csv(output_path)    # TODO: maybe parquet file instead ?? analyse it later

    return test_result


if __name__ == '__main__':

    # Initialize Spark Session
    spark_session = SparkSession.builder.appName("IDS for LoRaWAN").getOrCreate()

    # List of tasks (each task processes a different category of LoRaWAN messages according to the gateway)
    tasks = [(DatasetType.RXPK), (DatasetType.TXPK)]

    # Execute tasks in parallel using threads that use the Spark Session to process LoRaWAN data
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(execute, task, spark_session) for task in tasks]

        # Aguarda a conclusão de todas as tarefas
        for future in futures:
            print(future.result())
            


    # Stop the Spark Session
    spark_session.stop()