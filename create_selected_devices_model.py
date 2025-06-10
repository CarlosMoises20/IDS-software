
import argparse
from pyspark.sql.types import *
from common.spark_functions import create_spark_session
from generate_input_datasets import generate_input_datasets
from processing.message_classification import MessageClassification


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Message Classification for a specific DevAddr')
    parser.add_argument('--dev_addr', type=str, nargs='+', required=True, help='List of DevAddr to filter by')
    parser.add_argument('--datasets_format', type=str, choices=['json', 'parquet'], default='json',
                        help='Format of datasets to use (json or parquet)')
    parser.add_argument('--skip_dataset_generation_if_exists', type=str, choices=['True', 'False'], default='True',
                    help='Whether to skip model generation if it already exists')
    args = parser.parse_args()
    dev_addr_list = args.dev_addr
    datasets_format = args.datasets_format.lower()
    skipIfExists = args.skip_dataset_generation_if_exists == 'True'
    
    # Initialize Spark Session
    spark_session = create_spark_session()
    
    # If you want to see spark errors in debug level on console during the script running
    #spark_session.sparkContext.setLogLevel("DEBUG")

    generate_input_datasets(spark_session, datasets_format, skipIfExists)

    mc = MessageClassification(spark_session)

    mc.create_ml_models(dev_addr_list, datasets_format)

    spark_session.stop()

