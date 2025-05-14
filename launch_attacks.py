
## This script modifies the generated test dataset with introduced attacks, manipulating some parameters such as RSSI and LSNR

import argparse
from generate_input_datasets import generate_input_datasets
from common.spark_functions import create_spark_session
from common.dataset_type import DatasetType
from pyspark.sql.functions import col, lit, monotonically_increasing_id, row_number
from pyspark.sql import Window


def modify_parameters(spark_session, file_path, avg_num_samples_per_device, dev_addr_list, params, target_value):

    # Load the whole DataFrame
    df = spark_session.read.json(file_path)

    # Filter only DevAddrs of interest
    df_filtered = df.filter(col("DevAddr").isin(dev_addr_list))

    # Add row numbers per DevAddr
    window_spec = Window.partitionBy("DevAddr").orderBy(monotonically_increasing_id())
    df_with_rownum = df_filtered.withColumn("row_number", row_number().over(window_spec))

    # Mark only the first N rows per DevAddr
    df_to_modify = df_with_rownum.filter(col("row_number") <= avg_num_samples_per_device)

    # Apply modifications
    for param in params:
        df_to_modify = df_to_modify.withColumn(param, lit(target_value))

    # Drop the row_number helper column
    df_to_modify = df_to_modify.drop("row_number")

    # Combine modified + unmodified parts
    df_unmodified = df.join(df_to_modify.select("tmst"), on="tmst", how="left_anti")
    df_final = df_unmodified.unionByName(df_to_modify)

    # Overwrite file properly using Spark
    df_final.coalesce(1).write.mode("overwrite").json(file_path)

    print(f"✅ File {file_path} successfully modified")



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
                      avg_num_samples_per_device=1, 
                      dev_addr_list=["0000AB43", "0000A65E"],
                      params=["rssi", "lsnr1"], 
                      target_value=4000)
    
    """modify_parameters(spark_session=spark_session,
                      file_path=txpk_dataset_path, 
                      avg_num_samples_per_device=1, 
                      dev_addr_list=["0000AB43", "0000A65E"],
                      params=["rssi", "lsnr1"], 
                      target_value=4000)"""

    spark_session.stop()