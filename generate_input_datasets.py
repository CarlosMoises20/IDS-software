
## This script generates the dataset used to load all LoRaWAN messages to train and test ML models

import time, os, argparse, glob
from pathlib import Path
from common.spark_functions import create_spark_session
from common.dataset_type import DatasetType
from prepareData.prepareData import pre_process_type
from common.auxiliary_functions import format_time



"""
This function generates input datasets, already with pre-processing applied to prepare data for model training
It generates one dataset for RXPK messages and other for TXPK messages

"""
def generate_input_datasets(spark_session, format, skipIfExists=False):

    start_time = time.time()

    for dataset_type in [key for key in DatasetType]:

        dataset_name = dataset_type.value["name"].lower()

        dataset_path = f'./generatedDatasets/{dataset_name}/lorawan_dataset.{format}'

        if os.path.exists(dataset_path) and skipIfExists:
            print(f'{dataset_name.upper()} dataset in {format.upper()} format already exists')

        else:

            # Use glob to safely expand the wildcard
            pattern = str(Path(f"./datasets/{dataset_name}_*.log").resolve())
            input_files = glob.glob(pattern)

            # Load JSON files using expanded list
            df = spark_session.read.json(input_files).cache()

            # Pre-process and split
            df = pre_process_type(df, dataset_type)

            # Write datasets
            if format == "json":
                df.write.mode("overwrite").json(dataset_path)
            else:
                df.write.mode("overwrite").parquet(dataset_path)
                
            print(f'{dataset_name.upper()} dataset in {format.upper()} format was generated')


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

    generate_input_datasets(spark_session, datasets_format, skipIfExists=True)

    spark_session.stop()