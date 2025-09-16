
## All constants are defined here

import os, random

'''SPARK'''
SPARK_APP_NAME = "IDS for LoRaWAN network"
SPARK_PORT = "4060"
SPARK_EXECUTOR_MEMORY = "20g"
SPARK_DRIVER_MEMORY = "20g"
SPARK_EXECUTOR_MEMORY_OVERHEAD = "4g"
SPARK_EXECUTOR_CORES = os.cpu_count()
SPARK_NETWORK_TIMEOUT = "180s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = "30s"
SPARK_PRE_PROCESSING_NUM_PARTITIONS = "70" 
SPARK_PROCESSING_NUM_PARTITIONS = "30" 
SPARK_FILES_MAX_PARTITION_BYTES = "134217728"
SPARK_AUTO_BROADCAST_JOIN_THRESHOLD = "-1"                        
SPARK_SERIALIZER = "org.apache.spark.serializer.KryoSerializer"     
SPARK_SQL_ANSI_ENABLED = "false"

'''UDP SOCKET'''
UDP_IP = "0.0.0.0"
UDP_PORT = 5200

'''KAFKA VARIABLES'''
KAFKA_PORT = 9092

'''SPARK JAR FILES'''
SPARK_JARS = [
    os.path.join(os.environ.get("SPARK_HOME"), "jars", "isolation-forest_3.3.2_2.12-4.0.1.jar"),
    os.path.join(os.environ.get("SPARK_HOME"), "jars", "spark-sql-kafka-0-10_2.12-3.3.2.jar"),
    os.path.join(os.environ.get("SPARK_HOME"), "jars", "kafka-clients-2.8.0.jar"),
    os.path.join(os.environ.get("SPARK_HOME"), "jars", "spark-token-provider-kafka-0-10_2.12-3.3.2.jar"),
    os.path.join(os.environ.get("SPARK_HOME"), "jars", "commons-pool2-2.12.1.jar")
]


'''LORAWAN PARAMETERS VALUES'''
SF_LIST = [7, 8, 9, 10, 11, 12]
BW_LIST = [125, 250, 500]
PHY_PAYLOAD_LEN_LIST_ABNORMAL_VALUES = [random.randint(200, 10000) for _ in range(5)]
