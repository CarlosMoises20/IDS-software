
from crate.client import connect
from pyspark.sql import SparkSession
from message_classification import MessageClassification as mc
from dataset_type import DatasetType
from constants import DATABASE_URI


# TODO: implement a version that receives new messages in real time (stream processing), using the models ('results')
# retrieved from CrateDB



if __name__ == '__main__':

    # Initialize Spark Session
    spark_session = SparkSession.builder \
                                .appName("IDS for LoRaWAN network") \
                                .config("spark.sql.shuffle.partitions", "200")  \
                                .config("spark.sql.autoBroadcastJoinThreshold", "-1")  \
                                .config("spark.sql.files.maxPartitionBytes", "134217728")  \
                                .config("spark.executor.memory", "4g") \
                                .config("spark.driver.memory", "4g") \
                                .getOrCreate()
    
    # Connect with database (CrateDB)
    db_connection = connect(DATABASE_URI)
    cursor = db_connection.cursor()


    # TODO: use database connection to retrieve 'RXPK' and 'TXPK' models
        # 1 - reads the message
        # 2 - determines if message is 'rxpk' or 'txpk', and based on that, define dataset_type for that message
        # 3 - the dataset_type will determine the model that will be used to process the message and also the type of processing performed
        # 4 - convert the message to a dataframe
        # 5 - apply pre-processing using the corresponding class (RxpkPreProcessing or TxpkPreProcessing)
        # 6 - classify the message using the corresponding model retrieved from database