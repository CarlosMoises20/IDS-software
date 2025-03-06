
from auxiliaryFunctions.general import bind_dir_files
import os


class MessageClassification:

    """
    Function to execute the IDS

    It receives the spark session (spark_session) that handles the dataset processing and
    the corresponding dataset type (dataset_type) defined by DatasetType Enum

    It returns the processing results, namely the accuracy and the confusion matrix that show the
    model performance

    """
    @staticmethod
    def message_classification(spark_session, dataset_type):

        # Define the dataset root path
        dataset_root_path = "./datasets"
        
        ### Bind all log files into a single log file if it doesn't exist yet,
        ### to simplify data processing
        combined_logs_filename = bind_dir_files(dataset_root_path, dataset_type)

        # Load the dataset into a Spark Dataframe
        df = spark_session.read.json(combined_logs_filename)

        ### Initialize pre-processing class
        pre_processing = dataset_type.value["pre_processing_class"]

        # Call pre-processing method on the corresponding class
        df = pre_processing.pre_process_data(df)

        # randomly divide dataset into training (70%) and test (30%)
        # and set a seed in order to ensure reproducibility, which is important to 
        # ensure that the model is always trained and tested on the same examples each time the
        # model is run. This is important to compare the model's performance in different situations
        # (this proportion can be modified according to the results)
        df_train, df_test = df.randomSplit([0.7, 0.3], seed=522)

        ### Initialize processing class
        processing = dataset_type.value["processing_class"]

        # Call processing method on the corresponding class and return the processing results
        return processing.process_data(df_train, df_test)

        
