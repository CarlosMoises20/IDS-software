
## All constants are defined here


'''SPARK'''
SPARK_APP_NAME = "IDS for LoRaWAN network"
SPARK_PORT = "4050"
SPARK_EXECUTOR_MEMORY = "4g"  # Reduced to 4 GB for local machine
SPARK_DRIVER_MEMORY = "4g"  # Reduced to 4 GB for local machine
SPARK_EXECUTOR_MEMORY_OVERHEAD = "2g"  # Increased memory overhead for more headroom
SPARK_NETWORK_TIMEOUT = "300s"  # Reduced timeout for faster error detection
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = "60s"
SPARK_NUM_PARTITIONS = "150"  # Adjusted number of partitions based on data size
SPARK_FILES_MAX_PARTITION_BYTES = "268435456"


'''CRATEDB'''
CRATEDB_HOST = 'localhost'
CRATEDB_PORT = 4201
CRATEDB_URI = f'http://{CRATEDB_HOST}:{CRATEDB_PORT}'


'''ANOMALY DETECTION'''
SF_LIST = [7, 8, 9, 10, 11, 12]
CODR_LIST = [1, 2, 3, 4]
LSNR_MIN = -20
LSNR_MAX = 10
RSSI_MIN = -130
RSSI_MAX = -10
TMST_MIN = 2000     # miliseconds
LEN_MIN = 5         # REVIEW
LEN_MAX = 20        # REVIEW


# Expected frequency values (LoRaWAN operates on specific frequencies)
EXPECTED_FREQUENCIES = [868.1, 868.3, 868.5, 868.7, 868.9]  # EU868 example

# Maximum allowed timestamp difference (adjust based on real data)
EXPECTED_TMST_THRESHOLD = 1000000  # Example threshold (microseconds)
