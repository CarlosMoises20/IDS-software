
from pyspark.sql import SparkSession
from processing.message_classification import MessageClassification
from common.constants import *
from common.spark_functions import create_spark_session
from common.input_dataset_format import DatasetFormatType
from generate_input_datasets import generate_input_datasets


if __name__ == '__main__':

    spark_session = create_spark_session()

    generate_input_datasets(spark_session, DatasetFormatType.JSON)

    # Initialize class used for network intrusion detection
    mc = MessageClassification(spark_session)

    mc.classify_new_incoming_messages()