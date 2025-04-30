
## All constants are defined here

import os

'''SPARK'''
SPARK_APP_NAME = "IDS for LoRaWAN network"
SPARK_PORT = "4050"
SPARK_EXECUTOR_MEMORY = "20g"
SPARK_DRIVER_MEMORY = "20g"
SPARK_EXECUTOR_MEMORY_OVERHEAD = "4g"
SPARK_EXECUTOR_CORES = os.cpu_count()
SPARK_NETWORK_TIMEOUT = "180s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = "30s"
SPARK_PRE_PROCESSING_NUM_PARTITIONS = "250" 
SPARK_PROCESSING_NUM_PARTITIONS = "30" 
SPARK_FILES_MAX_PARTITION_BYTES = "134217728"
SPARK_AUTO_BROADCAST_JOIN_THRESHOLD = "-1"                          # best approach to handle the big weight of Deep-Learning processing
SPARK_SERIALIZER = "org.apache.spark.serializer.KryoSerializer"     # If using MLlib or saving/loading models
SPARK_KNN_PACKAGE = "com.jelmerk:hnswlib-pyspark_2.12:2.0.0-beta-1"