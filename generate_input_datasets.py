
## This script generates the dataset used to load all LoRaWAN messages to train and test ML models

import time
from common.auxiliary_functions import bind_dir_files, format_time, create_spark_session
from common.dataset_type import DatasetType
from common.constants import *
from pyspark.sql import SparkSession


start_time = time.time()

spark_session = create_spark_session()

filename = bind_dir_files(spark_session=spark_session, dataset_type=DatasetType) 

spark_session.stop()

end_time = time.time()

print(f"Time of generation (or not) of file '{filename}':", format_time(end_time - start_time))