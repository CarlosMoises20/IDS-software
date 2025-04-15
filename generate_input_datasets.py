
## This script generates the datasets used to load all LoRaWAN messages to train and test ML models

import time
from common.auxiliary_functions import bind_dir_files, format_time
from common.dataset_type import DatasetType
from common.constants import *
from pyspark.sql import SparkSession


start_time = time.time()

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

filename_rxpk, filename_txpk = (bind_dir_files(spark_session=spark_session, 
                                               dataset_type=dataset_type) for dataset_type in [key for key in list(DatasetType)])

spark_session.stop()

end_time = time.time()

print(f"Time of generation (or not) of files '{filename_rxpk}' and '{filename_txpk}':",
      format_time(end_time - start_time))