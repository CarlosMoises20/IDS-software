
from pyspark.sql.functions import col, when, count, explode, expr
from crate.client import connect
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from xgboost.spark import SparkXGBClassifier, SparkXGBRegressor
from preProcessing.pre_processing import *
from auxiliaryFunctions.general_functions import *
from auxiliaryFunctions.ids_functions import *
from dataset_type import DatasetType
from preProcessing.pre_processing import PreProcessing



class DataProcessing:

    def __init__(self, spark_session, dataset, dataset_type):
        self.__spark_session = spark_session
        self.__dataset = dataset
        self.__dataset_type = dataset_type


    # TODO: finish or remove if unnecessary
    def __process_message(message_type):

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


    """
    Function to prepare each one of the datasets

    It receives the spark session (spark_session) that handles the dataset processing, the dataset itself (dataset) and
    the corresponding dataset type (dataset_type) in an Enum

    It returns the dataframe split in training and test dataframes, and a list of strings that contains
    all the different message types from the dataset

    """
    def __prepare_dataset(self):
        
        combined_logs_filename = './combinedDatasets/combined_rxpk_logs.log' if self.__dataset_type == DatasetType.RXPK \
                else './combinedDatasets/combined_txpk_logs.log'
        
        ### Bind all log files into a single log file if it doesn't exist yet,
        ### to simplify data processing
        bind_dir_files(self.__dataset, combined_logs_filename)

        # Load the dataset into a Spark Dataframe
        df = self.__spark_session.read.json(combined_logs_filename)

        ### Pre-Processing
        pre_processing = PreProcessing(df)

        df = pre_processing.pre_process_rxpk_dataset() if self.__dataset_type == DatasetType.RXPK \
            else pre_processing.pre_process_txpk_dataset()

        # Get all the different message types of the dataset
        message_types = df.select("MessageType").distinct().rdd.flatMap(lambda x: x).collect()

        # randomly divide dataset into training (2/3) and test (1/3)
        df_train, df_test = df.randomSplit([2/3, 1/3])  

        return message_types, df_train, df_test



    # TODO: this should be an abstract method to implement in two separate classes, one dedicated for 'txpk' and one for 'rxpk'
    def process_dataset(self):

        
        message_types, df_train, df_test = self.__prepare_dataset()
        
        # TODO: continue

        # Possible approach for numeric attributes
        assembler = VectorAssembler(inputCols=get_numeric_attributes(df_train.schema), outputCol="features", handleInvalid="keep")


        # separate "rxpk" and "txpk" logic


        # TODO: study approach for categorical attributes or an approach for all types of attributes

        #rf = RandomForestClassifier(numTrees=5, maxDepth=4)

        #rf_model = rf.fit(df_train)

        #rf_predictions = rf_model.transform(df_test)

        # TODO: calculate "RFU", it comes from various attributes


        pass




