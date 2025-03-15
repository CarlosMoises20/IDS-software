
from crate.client import connect
from pyspark.sql import SparkSession
from message_classification import MessageClassification as mc
from dataset_type import DatasetType
from constants import *


# TODO: implement a version that receives new messages in real time (stream processing), using the models ('results')
# retrieved from CrateDB



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
    
    # Connect with database (CrateDB)
    db_connection = connect(CRATEDB_URI)
    cursor = db_connection.cursor()


    # TODO: run the application, opening a connection to receive messages in real-time, that only ends when requested


    # TODO: use database connection to retrieve 'RXPK' and 'TXPK' models
        # 1 - reads the message
        # 2 - determines if message is 'rxpk' or 'txpk', and based on that, define dataset_type for that message
        # 3 - the dataset_type will determine the model that will be used to process the message and also the type of processing performed
        # 4 - convert the message to a dataframe
        # 5 - apply pre-processing using the corresponding class (RxpkPreProcessing or TxpkPreProcessing)
        # 6 - classify the message using the corresponding model retrieved from database