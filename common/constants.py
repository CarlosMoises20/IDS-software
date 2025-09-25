
## All constants are defined here

import os, random

'''SPARK'''
SPARK_APP_NAME = "IDS for LoRaWAN network"
SPARK_PORT = "4060"
SPARK_MASTER = "local[*,4]"
SPARK_APP_NAME = "IDS for LoRaWAN network"
SPARK_EXECUTOR_MEMORY = "40g"                  # Memory available for the Spark executor
SPARK_DRIVER_MEMORY = "40g"                    # Memory available for the Spark driver
SPARK_EXECUTOR_CORES = os.cpu_count() - 1      # Number of cores for Spark is set as the number of CPU cores of local machine, to speed up processing
SPARK_PROCESSING_NUM_PARTITIONS = "2500"       # Number of Spark partitions for parallel processing
SPARK_AUTO_BROADCAST_JOIN_THRESHOLD = "-1"     # By setting this to -1, broadcasting is disabled, removing the maximum size of a table broadcasted to all worker nodes when performing a join; this speeds up performance

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
PHY_PAYLOAD_LEN_LIST_ABNORMAL_VALUES = [random.randint(200, 10000) for _ in range(7)]
RSSI_ESTIMATED_INTERVAL = [round(random.uniform(-130, -20), 1) for _ in range(7)]
LSNR_ESTIMATED_INTERVAL = [round(random.uniform(-20, -10), 1) for _ in range(7)]
CODR_ESTIMATED_INTERVAL = [round(random.uniform(0, 1), 1) for _ in range(5)]
RFCH_VALUES = [random.randint(0, 30) for _ in range(7)]
FCTRL_VALUES = [random.randint(0, 255) for _ in range(10)]
MHDR_VALUES = [random.randint(0, 255) for _ in range(10)]
MIC_VALUES = [random.randint(0, 4294967296) for _ in range(10)]