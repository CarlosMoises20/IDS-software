
## This script generates the dataset used to load all LoRaWAN messages to train and test ML models

import time, os, argparse
from common.spark_functions import create_spark_session, sample_random_split
from common.dataset_type import DatasetType
from prepareData.prepareData import pre_process_type
from common.auxiliary_functions import format_time



"""
This function generates input datasets, already with pre-processing applied to prepare data for model training

"""
def generate_input_datasets(spark_session, format):

    for dataset_type in [key for key in DatasetType]:

        if os.path.exists(f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_train.{format}') and \
            os.path.exists(f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_test.{format}'):

            print("Input datasets for", dataset_type.value["name"].upper(), "in format", format.upper(), 
                  "already exist. Skipping generation")

        else:

            df = spark_session.read.json(f'./datasets/{dataset_type.value["name"]}_*.log')

            df = df.cache()

            df = pre_process_type(df, dataset_type)

            df_train, df_test = sample_random_split(df)

            if format == "json":

                df_train.write.mode("overwrite").json(f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_train.json')
                
                df_test.write.mode("overwrite").json(f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_test.json')

            if format == "parquet":

                df_train.write.mode("overwrite").parquet(f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_train.parquet')
                
                df_test.write.mode("overwrite").parquet(f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_test.parquet')


            print(f'{format.upper()} files for {dataset_type.value["name"].upper()} generated')


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate input datasets')
    parser.add_argument('--datasets_format', type=str, choices=['json', 'parquet'], default='json',
                        help='Format of datasets to use (json or parquet)')
    args = parser.parse_args()
    datasets_format = args.datasets_format.lower()

    start_time = time.time()

    spark_session = create_spark_session()

    generate_input_datasets(spark_session, datasets_format)

    spark_session.stop()

    end_time = time.time()

    print("Total time of training and testing files generation with pre-processing included:", 
          format_time(end_time - start_time))
