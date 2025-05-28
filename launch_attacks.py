
## This script modifies the generated test dataset with introduced attacks, manipulating some parameters such as RSSI and LSNR

import argparse, time, random
from generate_input_datasets import generate_input_datasets
from common.auxiliary_functions import format_time
from common.spark_functions import create_spark_session
from common.dataset_type import DatasetType
from pyspark.sql.functions import col, lit
from pyspark.sql import Window


def modify_parameters(spark_session, file_path, dev_addr_list, params, target_values, dataset_format):

    start_time = time.time()

    # Load the dataset
    if dataset_format == "json":
        df = spark_session.read.json(file_path)
    else:
        df = spark_session.read.parquet(file_path)

    # Filter only rows with DevAddrs of interest
    df_filtered = df.filter(col("DevAddr").isin(dev_addr_list))

    # Get 25% random sample from each DevAddr
    fraction_per_dev = {addr: 0.25 for addr in dev_addr_list}
    df_to_modify = df_filtered.sampleBy("DevAddr", fractions=fraction_per_dev, seed=42)

    # Modify selected rows
    for param in params:
        random_value = random.choice(target_values)
        df_to_modify = df_to_modify.withColumn(param, lit(random_value))

    df_to_modify = df_to_modify.withColumn("intrusion", lit(1))

    # Filter out modified rows from original DataFrame
    df_unmodified = df.join(df_to_modify.select("tmst"), on="tmst", how="left_anti")

    # Combine modified and unmodified
    df_final = df_unmodified.unionByName(df_to_modify)

    # Save result
    if dataset_format == "json":
        df_final.coalesce(1).write.mode("overwrite").json(file_path)
    else:
        df_final.coalesce(1).write.mode("overwrite").parquet(file_path)

    end_time = time.time()
    print(f"File {file_path} successfully modified in {format_time(end_time - start_time)}")



if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate input datasets')
    parser.add_argument('--datasets_format', type=str, choices=['json', 'parquet'], default='json',
                        help='Format of datasets to use (json or parquet)')
    args = parser.parse_args()
    datasets_format = args.datasets_format.lower()

    spark_session = create_spark_session()

    generate_input_datasets(spark_session, datasets_format)

    rxpk_dataset_path, txpk_dataset_path = (f"./generatedDatasets/{dataset_type}/lorawan_dataset_test.{datasets_format}"
                                                for dataset_type in [key.value["name"] for key in DatasetType])

    modify_parameters(spark_session=spark_session,
                      file_path=rxpk_dataset_path,  
                      dev_addr_list=["26002285", "26002E44", "26012B8C", "0000AB43", 
                                     "0000A65E", "0000BF53"],
                      params=["rssi"], 
                      target_values=[-250, 100],
                      dataset_format=datasets_format)
    
    modify_parameters(spark_session=spark_session,
                      file_path=txpk_dataset_path, 
                      dev_addr_list=["26002285", "26002E44", "26012B8C"],
                      params=["dataLen"], 
                      target_values=[400, 2, 350],
                      dataset_format=datasets_format)

    spark_session.stop()