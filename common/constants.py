
## All constants are defined here

import os, random

'''SPARK'''
SPARK_APP_NAME = "IDS for LoRaWAN network"
SPARK_PORT = "4060"
SPARK_EXECUTOR_MEMORY = "12g"                   # Memory available for the Spark executor
SPARK_DRIVER_MEMORY = "12g"                     # Memory available for the Spark driver
SPARK_EXECUTOR_CORES = os.cpu_count() - 1      # Number of cores for Spark is set as the number of CPU cores of local machine, to speed up processing
SPARK_PROCESSING_NUM_PARTITIONS = "1000"        # Number of Spark partitions for parallel processing
SPARK_AUTO_BROADCAST_JOIN_THRESHOLD = "-1"      # By setting this to -1, broadcasting is disabled, removing the maximum size of a table broadcasted to all worker nodes when performing a join; this speeds up performance                           

'''SPARK JAR FILES'''
SPARK_JARS = [
    os.path.join(os.environ.get("SPARK_HOME"), "jars", "isolation-forest_3.3.2_2.12-4.0.6.jar"),
    os.path.join(os.environ.get("SPARK_HOME"), "jars", "spark-sql-kafka-0-10_2.12-3.3.2.jar"),
    os.path.join(os.environ.get("SPARK_HOME"), "jars", "kafka-clients-2.8.0.jar"),
    os.path.join(os.environ.get("SPARK_HOME"), "jars", "spark-token-provider-kafka-0-10_2.12-3.3.2.jar"),
    os.path.join(os.environ.get("SPARK_HOME"), "jars", "commons-pool2-2.12.1.jar")
]

'''UDP SOCKET'''
UDP_IP = "0.0.0.0"
UDP_PORT = 5200

'''KAFKA VARIABLES'''
KAFKA_PORT = 9092

'''LORAWAN PARAMETERS VALUES'''
SF_LIST = [7, 8, 9, 10, 11, 12]
BW_LIST = [125, 250, 500]
PHY_PAYLOAD_LEN_LIST_ABNORMAL_VALUES = [random.randint(1500, 10000) for _ in range(5)]
