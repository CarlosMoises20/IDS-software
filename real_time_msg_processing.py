
import argparse
from processing.message_classification import MessageClassification
from common.constants import *
from common.spark_functions import create_spark_session
from generate_input_datasets import generate_input_datasets


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate input datasets')
    parser.add_argument('--datasets_format', type=str, choices=['json', 'parquet'], default='json',
                        help='Format of datasets to use (json or parquet)')
    args = parser.parse_args()
    datasets_format = args.datasets_format.lower()

    spark_session = create_spark_session()

    generate_input_datasets(spark_session, datasets_format)

    # Initialize class used for network intrusion detection
    mc = MessageClassification(spark_session)

    mc.classify_new_incoming_messages()