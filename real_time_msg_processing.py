
import argparse
from processing.message_classification import MessageClassification
from common.constants import *
from common.spark_functions import create_spark_session
from generate_input_datasets import generate_input_datasets


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate input datasets')
    parser.add_argument('--datasets_format', type=str, choices=['json', 'parquet'], default='parquet',
                        help='Format of datasets to use (json or parquet)')
    parser.add_argument('--skip_dataset_generation_if_exists', type=str, choices=['True', 'False'], default='True',
                    help='Whether to skip model generation if it already exists')
    args = parser.parse_args()
    datasets_format = args.datasets_format.lower()
    skipIfExists = (args.skip_dataset_generation_if_exists == 'True')

    spark_session = create_spark_session()

    # If you want to see spark errors in debug level on console during script execution
    #spark_session.sparkContext.setLogLevel("DEBUG")

    generate_input_datasets(spark_session, datasets_format, skipIfExists)

    # Initialize class used for network intrusion detection
    mc = MessageClassification(spark_session)

    mc.classify_new_incoming_messages(datasets_format)