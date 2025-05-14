
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

    start_time = time.time()

    generated = False

    for dataset_type in [key for key in DatasetType]:

        train_dataset_path = f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_train.{format}'
        test_dataset_path = f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_test.{format}'

        # This condition avoids consuming time on reading the dataset and pre-processing to generate new datasets, 
        # if both datasets already exist 
        if os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path):

            print("Input training dataset for", dataset_type.value["name"].upper(), "in format", format.upper(), 
                  "already exist. Skipping generation")

        else:

            df = spark_session.read.json(f'./datasets/{dataset_type.value["name"]}_*.log')

            df = df.cache()

            df = pre_process_type(df, dataset_type)

            df_train, df_test = sample_random_split(df)

            if format == "json":

                if not os.path.exists(train_dataset_path):
                    df_train.write.mode("overwrite").json(train_dataset_path)

                else:
                    print(f'Train {dataset_type.value["name"].upper()} dataset in {format.upper()} format already exists')
                
                if not os.path.exists(test_dataset_path):
                    df_test.write.mode("overwrite").json(test_dataset_path)

                else:
                    print(f'Test {dataset_type.value["name"].upper()} dataset in {format.upper()} format already exists')

            if format == "parquet":

                if not os.path.exists(train_dataset_path):
                    df_train.write.mode("overwrite").parquet(train_dataset_path)

                else:
                    print(f'Train {dataset_type.value["name"].upper()} dataset in {format.upper()} format already exists')

                if not os.path.exists(test_dataset_path):
                    df_test.write.mode("overwrite").parquet(test_dataset_path)

                else:
                    print(f'Test {dataset_type.value["name"].upper()} dataset in {format.upper()} format already exists')

            print(f'{format.upper()} files for {dataset_type.value["name"].upper()} generated')

            generated = True

    end_time = time.time()

    if generated:
    
        print("Total time of generation of training and testing files with pre-processing included:", 
            format_time(end_time - start_time))


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate input datasets')
    parser.add_argument('--datasets_format', type=str, choices=['json', 'parquet'], default='json',
                        help='Format of datasets to use (json or parquet)')
    args = parser.parse_args()
    datasets_format = args.datasets_format.lower()

    spark_session = create_spark_session()

    generate_input_datasets(spark_session, datasets_format)

    spark_session.stop()