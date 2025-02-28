
from pyspark.sql.functions import col, when, count, explode, expr
from crate.client import connect
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from xgboost.spark import SparkXGBClassifier, SparkXGBRegressor
from intrusion_detection import *
from data_pre_processing import *
from auxiliary_functions import *




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


    # randomly divide dataset into training (2/3) and test (1/3)
    df_train, df_test = df.randomSplit([2/3, 1/3])
    
    # TODO: continue

    # Possible approach for numeric attributes
    assembler = VectorAssembler(inputCols=get_numeric_attributes(df.schema), outputCol="features", handleInvalid="keep")

    # TODO: study approach for categorical attributes or an approach for all types of attributes

    #rf = RandomForestClassifier(numTrees=5, maxDepth=4)

    #rf_model = rf.fit(df_train)

    #rf_predictions = rf_model.transform(df_test)

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

    # randomly divide dataset into training (2/3) and test (1/3)
    df_train, df_test = df.randomSplit([2/3, 1/3])

    # TODO: continue

    pass


