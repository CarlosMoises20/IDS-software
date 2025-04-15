
from crate.client import connect
from pyspark.sql import SparkSession
from processing.message_classification import MessageClassification
from common.constants import *


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


    # Initialize class used for network intrusion detection
    mc = MessageClassification(spark_session)

    # TODO: implement a function that classifies new messages in real time based on the corresponding models, probably in this way
        # 1 - reads the message
        # 2 - converts the message to a dataframe
        # 3 - verifies the schema to check if its "RXPK" or "TXPK"
        # 5 - apply pre-processing using the corresponding class (RxpkPreProcessing or TxpkPreProcessing)
        # 6 - classify the message using the corresponding model retrieved from MLFlow, based on DevAddr