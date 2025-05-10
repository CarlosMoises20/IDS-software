
## This script generates the dataset used to load all LoRaWAN messages to train and test ML models

import time, os
from common.spark_functions import create_spark_session, sample_random_split
from common.dataset_type import DatasetType
from prepareData.prepareData import pre_process_type
from common.auxiliary_functions import format_time
from common.input_dataset_format import DatasetFormatType



"""
This function generates input datasets, already with pre-processing applied to prepare data for model training

"""
def generate_input_datasets(spark_session, format=DatasetFormatType.PARQUET):

    for dataset_type in [key for key in DatasetType]:

        if os.path.exists(f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_train.{format.value}') and \
            os.path.exists(f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_test.{format.value}'):

            print("Input datasets for", dataset_type.value["name"], "in format", format.value, 
                  "already exist. Skipping generation")

        else:

            df = spark_session.read.json(f'./datasets/{dataset_type.value["name"]}_*.log')

            df = df.cache()

            df = pre_process_type(df, dataset_type)

            df_train, df_test = sample_random_split(df_model=df, seed=422)

            if format == DatasetFormatType.JSON:

                df_train.write.mode("overwrite").json(f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_train.{format.value}')
                
                df_test.write.mode("overwrite").json(f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_test.{format.value}')

            if format == DatasetFormatType.PARQUET:

                df_train.write.mode("overwrite").parquet(f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_train.{format.value}')
                
                df_test.write.mode("overwrite").parquet(f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_test.{format.value}')


            print(f'{format.value} files for {dataset_type.value["name"]} generated')


if __name__ == '__main__':

    start_time = time.time()

    spark_session = create_spark_session()

    generate_input_datasets(spark_session, format=DatasetFormatType.JSON)

    spark_session.stop()

    end_time = time.time()

    print("Total time of training and testing files generation with pre-processing included:", 
          format_time(end_time - start_time))
