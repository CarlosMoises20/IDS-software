
## This script generates the dataset used to load all LoRaWAN messages to train and test ML models

import time, os, argparse, glob
from pathlib import Path
from common.spark_functions import create_spark_session, train_test_split
from common.dataset_type import DatasetType
from prepareData.prepareData import pre_process_type
from common.auxiliary_functions import format_time



"""
This function generates input datasets, already with pre-processing applied to prepare data for model training

"""
def generate_input_datasets(spark_session, format):

    start_time = time.time()

    for dataset_type in [key for key in DatasetType]:

        dataset_name = dataset_type.value["name"].lower()

        train_dataset_path = f'./generatedDatasets/{dataset_name}/lorawan_dataset_train.{format}'
        test_dataset_path = f'./generatedDatasets/{dataset_name}/lorawan_dataset_test.{format}'

        # Use glob to safely expand the wildcard
        pattern = str(Path(f"./datasets/{dataset_name}_*.log").resolve())
        input_files = glob.glob(pattern)

        # Load JSON files using expanded list
        df = spark_session.read.json(input_files).cache()

        # Pre-process and split
        df = pre_process_type(df, dataset_type)
        df_train, df_test = train_test_split(df)

        # Write datasets
        if format == "json":
            if not os.path.exists(train_dataset_path):
                df_train.write.mode("overwrite").json(train_dataset_path)
                print(f'Train {dataset_name.upper()} dataset in {format.upper()} was generated')
            else:
                print(f'Train {dataset_name.upper()} dataset in {format.upper()} format already exists')

            if not os.path.exists(test_dataset_path):
                df_test.write.mode("overwrite").json(test_dataset_path)
                print(f'Test {dataset_name.upper()} dataset in {format.upper()} format was generated')
            else:
                print(f'Test {dataset_name.upper()} dataset in {format.upper()} format already exists')

        elif format == "parquet":
            if not os.path.exists(train_dataset_path):
                df_train.write.mode("overwrite").parquet(train_dataset_path)
                print(f'Train {dataset_name.upper()} dataset in {format.upper()} format was generated')
            else:
                print(f'Train {dataset_name.upper()} dataset in {format.upper()} format already exists')

            if not os.path.exists(test_dataset_path):
                df_test.write.mode("overwrite").parquet(test_dataset_path)
                print(f'Test {dataset_name.upper()} dataset in {format.upper()} was generated')
            else:
                print(f'Test {dataset_name.upper()} dataset in {format.upper()} format already exists')

    end_time = time.time()
    print("Total time of execution of 'generate_input_datasets.py':",
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