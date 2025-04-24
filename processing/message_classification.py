
import time
from prepareData.prepareData import prepare_past_dataset
from common.auxiliary_functions import format_time
from common.constants import SPARK_PROCESSING_NUM_PARTITIONS
from models.model_utils import ModelUtils
from pyspark import SparkContext



class MessageClassification:

    def __init__(self, spark_session):
        self.__spark_session = spark_session


    """
    Define the function that Spark will run in parallel per device's DevAddr
    
        pdf: pandas dataframe

    """
    @staticmethod
    def model_train_udf(pdf):
        dev_addr = pdf["DevAddr"].iloc[0]
        modelclass = ModelUtils()
        modelclass.create_model(pdf, dev_addr) 
        return pdf


    """
    Function to execute the IDS

    It receives the spark session (spark_session) that handles the dataset processing and
    the corresponding dataset type (dataset_type) defined by DatasetType Enum

    It stores the models as artifacts using MLFlow, as well as their associated informations 
    such as metric evaluations and the associated DevAddr 

        dev_addr_list - an optional parameter to specify, as a list of strings, the DevAddr of the devices
                        from which the user pretends to create models; if not specified, models from all 
                        devices will be created

    """
    def create_ml_models(self, dev_addr_list=None):

        # pre-processing: prepare past dataset
        df = prepare_past_dataset(self.__spark_session)

        # Splits the dataframe into "SPARK_PROCESSING_NUM_PARTITIONS" partitions during pre-processing
        df = df.coalesce(numPartitions=int(SPARK_PROCESSING_NUM_PARTITIONS))

        ### Begin processing
        start_time = time.time()

        # When dev_addr_list is not specified, models of all devices are created
        if dev_addr_list is None:

            # Cast DevAddr column to integer and get distinct values
            dev_addr_list = df.select("DevAddr").filter(df["DevAddr"].isNotNull()).distinct()

            # Convert to a list of integers
            dev_addr_list = [row.DevAddr for row in dev_addr_list.collect()]

        else:
            # When dev_addr_list is specified, remove, from the dataset, rows 
            # whose DevAddr does not belong to the list defined by the user
            df = df.filter(df.DevAddr.isin(dev_addr_list))


        # create all models in parallel to accelerate code execution
        result_df = df.groupBy("DevAddr").applyInPandas(MessageClassification.model_train_udf, df.schema)


        # write the dataframe in a CSV file, excluding the column "features", because it's not necessary for visualization,
        # it's only used for ML algorithms' processing and its vector type is not supported by CSV
        # CSV has chosen since it's a simple-to-visualize and efficient format   
        result_df.drop("features", "features_dense") \
            .write \
            .mode("overwrite") \
            .option("header", True) \
            .csv("./generatedDatasets/ids_final_results")
    
        end_time = time.time()

        # Print the total time of pre-processing; the time is in seconds, minutes or hours
        print("Total time of processing:", format_time(end_time - start_time), "\n\n")

        return result_df