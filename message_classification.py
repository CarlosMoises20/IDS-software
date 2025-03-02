
from auxiliaryFunctions.general_functions import bind_dir_files
import os


class MessageClassification:

    # TODO: finish, or remove if unnecessary
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


    """
    Function to prepare each one of the datasets

    It receives the spark session (spark_session) that handles the dataset processing, the dataset itself (dataset) and
    the corresponding dataset type (dataset_type) in an Enum

    It returns the dataframe split in training and test dataframes, and a list of strings that contains
    all the different message types from the dataset

    """
    def message_classification(self, spark_session, dataset_type):

        # Define the dataset root path
        dataset_root_path = "./datasets"
        
        ### Bind all log files into a single log file if it doesn't exist yet,
        ### to simplify data processing
        combined_logs_filename = bind_dir_files(os.fsencode(dataset_root_path), dataset_type)

        # Load the dataset into a Spark Dataframe
        df = spark_session.read.json(combined_logs_filename)

        ### Initialize pre-processing class
        pre_processing = dataset_type.value["pre_processing_class"]

        # Call pre-processing method on the corresponding class
        df = pre_processing.pre_process_data(self, df)

        # Get all the different message types of the dataset
        message_types = df.select("MessageType").distinct().rdd.flatMap(lambda x: x).collect()

        # randomly divide dataset into training (2/3) and test (1/3)
        df_train, df_test = df.randomSplit([2/3, 1/3])  

        ### Initialize processing class
        processing = dataset_type.value["processing_class"]

        # Call processing method on the corresponding class
        processing.process_data(self, df_train, df_test, message_types)
