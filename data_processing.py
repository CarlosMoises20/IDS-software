
from pyspark.sql.functions import col, when, count, explode, expr
from crate.client import connect
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import NaiveBayes, BinaryRandomForestClassificationSummary
from pyspark.ml.feature import VectorAssembler
from xgboost.spark import SparkXGBClassifier, SparkXGBRegressor
from intrusion_detection import *
from data_pre_processing import *
from auxiliary_functions import *


### On this module, add functions, where each function process a different type of messages


# TODO: finish
def process_message(message_type):

    if (message_type == "Join Request"):
        pass
    elif (message_type == "Join Accept"):
        pass
    elif (message_type == "Unconfirmed Data Up"):
        pass
    elif (message_type == "Unconfirmed Data Down"):
        pass
    elif (message_type == "Confirmed Data Up"):
        pass
    elif (message_type == "Confirmed Data Down"):
        pass
    elif (message_type == "RFU"):
        pass
    elif (message_type == "Proprietary"):
        pass



def process_rxpk_dataset(spark_session, dataset):

    ### Bind all log files into a single log file if it doesn't exist yet,
    ### to simplify data processing
    
    combined_logs_filename = './combined_datasets/combined_rxpk_logs.log'
    bind_dir_files(dataset, combined_logs_filename)


    # Load the dataset into a Spark Dataframe
    df = spark_session.read.json(combined_logs_filename)


    ### Pre-Processing
    df = pre_process_rxpk_dataset(df)


    # divide dataset into training (2/3) and test (1/3)
    df_train, df_test = df.randomSplit([2/3, 1/3])


    feature_columns = ['rxpk.lsnr', 'rxpk.rssi', 'rxpk.freq', 'rxpk.size']
    
    assembler_rxpk = VectorAssembler(inputCols=feature_columns, outputCol="features")

    df_rxpk_train = assembler_rxpk.transform(df_train)
    
    # TODO: continue

    # TODO: calculate "RFU", it comes from various attributes


    pass




def process_txpk_dataset(spark_session, dataset):

    ### Bind all log files into a single log file if it doesn't exist yet
    ### to simplify data processing

    combined_logs_filename = './combined_datasets/combined_txpk_logs.log'
    bind_dir_files(dataset, combined_logs_filename)

    # Load the dataset into a Spark Dataframe
    df = spark_session.read.json(combined_logs_filename)

    ### Pre-Processing
    df = pre_process_txpk_dataset(df)

    # divide dataset into training (2/3) and test (1/3)
    df_train, df_test = df.randomSplit([2/3, 1/3])

    # TODO: continue

    pass


