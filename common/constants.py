
## All constants are defined here


'''SPARK'''
SPARK_APP_NAME = "IDS for LoRaWAN network"
SPARK_PORT = "4050"
SPARK_EXECUTOR_MEMORY = "20g"
SPARK_DRIVER_MEMORY = "16g"
SPARK_EXECUTOR_MEMORY_OVERHEAD = "4g"
SPARK_NETWORK_TIMEOUT = "180s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = "30s"
SPARK_NUM_PARTITIONS = "10" 
SPARK_FILES_MAX_PARTITION_BYTES = "134217728"


'''CRATEDB'''
CRATEDB_HOST = 'tfm-ct.dyn.fil.isel.pt'         # or another from your choice
CRATEDB_PORT = 4201                             # or another from your choice (it must be the same port of the container)
CRATEDB_URI = f'http://{CRATEDB_HOST}:{CRATEDB_PORT}'