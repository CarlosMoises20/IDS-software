
import argparse
from pyspark.sql.types import *
from common.spark_functions import create_spark_session
from generate_input_datasets import generate_input_datasets
from processing.message_classification import MessageClassification


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Message Classification for a specific DevAddr')
    
    parser.add_argument('--dev_addr', type=str, nargs='+', required=False, help='List of DevAddr to filter by')
    
    parser.add_argument('--datasets_format', type=str, choices=['json', 'parquet'], default='parquet',
                        help='Format of datasets to use (json or parquet)')
    
    parser.add_argument('--skip_dataset_generation_if_exists', type=str, choices=['True', 'False'], default='True',
                        help='Whether to skip model generation if it already exists')
    
    parser.add_argument('--ml_algorithm', type=str, 
                        choices=['lof', 'if_custom', 'if_sklearn', 'hbos', 'knn', 'ocsvm'], 
                        default='ocsvm',
                        help='ML Algorithm to choose to create the ML models')
    
    args = parser.parse_args()
    dev_addr_list = args.dev_addr
    datasets_format = args.datasets_format
    skip_if_exists = (args.skip_dataset_generation_if_exists == 'True')
    ml_algorithm = args.ml_algorithm
    
    # Initialize Spark Session
    spark_session = create_spark_session()
    
    # If you want to see spark errors in debug level on console during script execution
    #spark_session.sparkContext.setLogLevel("DEBUG")

    # Generate input datasets that will be used to create the models, if they don't exist yet
    generate_input_datasets(spark_session=spark_session, 
                            format=datasets_format, 
                            skip_if_exists=skip_if_exists)

    mc = MessageClassification(spark_session=spark_session,
                               ml_algorithm=ml_algorithm)

    mc.create_ml_models(dev_addr_list=dev_addr_list, 
                        datasets_format=datasets_format)

    spark_session.stop()

