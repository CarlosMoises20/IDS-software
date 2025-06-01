
import argparse
from pyspark.sql.types import *
from common.spark_functions import create_spark_session, modify_parameters
from generate_input_datasets import generate_input_datasets
from common.dataset_type import DatasetType
from models.functions import *
from processing.message_classification import MessageClassification


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Message Classification for a specific DevAddr')
    parser.add_argument('--dev_addr', type=str, nargs='+', required=True, help='List of DevAddr to filter by')
    parser.add_argument('--datasets_format', type=str, choices=['json', 'parquet'], default='json',
                        help='Format of datasets to use (json or parquet)')
    args = parser.parse_args()
    dev_addr_list = args.dev_addr
    datasets_format = args.datasets_format.lower()
    
    # Initialize Spark Session
    spark_session = create_spark_session()
    
    # If you want to see spark errors in debug level on console during the script running
    #spark_session.sparkContext.setLogLevel("DEBUG")

    generate_input_datasets(spark_session, datasets_format)

    rxpk_dataset_path, txpk_dataset_path = (f"./generatedDatasets/{dataset_type}/lorawan_dataset_test.{datasets_format}"
                                                for dataset_type in [key.value["name"] for key in DatasetType])

    sf_list = [1, 18, 19, 22, 23]
    bw_list = [250, 250, 500]
    len_list = [53, 55, 57, 109, 111]

    modify_parameters(spark_session=spark_session,
                      file_path=rxpk_dataset_path,  
                      dev_addr_list=dev_addr_list,
                      params=["SF", "BW", "payloadLen"], 
                      target_values=[sf_list, bw_list, len_list],
                      dataset_format=datasets_format)
    
    modify_parameters(spark_session=spark_session,
                      file_path=txpk_dataset_path, 
                      dev_addr_list=dev_addr_list,
                      params=["SF", "BW", "dataLen"], 
                      target_values=[sf_list, bw_list, len_list],
                      dataset_format=datasets_format)

    mc = MessageClassification(spark_session)

    mc.create_ml_models(dev_addr_list)

    spark_session.stop()

