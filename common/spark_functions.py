
import os, glob, time
from common.constants import *
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, lit, when, create_map
from pyspark.sql.types import StructType

"""
Auxiliary function to create spark session

"""
def create_spark_session():
    return SparkSession.builder \
                            .appName(SPARK_APP_NAME) \
                            .config("spark.ui.port", SPARK_PORT) \
                            .config("spark.sql.shuffle.partitions", SPARK_PRE_PROCESSING_NUM_PARTITIONS)  \
                            .config("spark.sql.files.maxPartitionBytes", SPARK_FILES_MAX_PARTITION_BYTES)  \
                            .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY) \
                            .config("spark.executor.cores", SPARK_EXECUTOR_CORES) \
                            .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
                            .config("spark.executor.memoryOverhead", SPARK_EXECUTOR_MEMORY_OVERHEAD) \
                            .config("spark.network.timeout", SPARK_NETWORK_TIMEOUT) \
                            .config("spark.executor.heartbeatInterval", SPARK_EXECUTOR_HEARTBEAT_INTERVAL) \
                            .config("spark.sql.autoBroadcastJoinThreshold", SPARK_AUTO_BROADCAST_JOIN_THRESHOLD) \
                            .config("spark.sql.ansi.enabled", SPARK_SQL_ANSI_ENABLED) \
                            .getOrCreate()
                            #.config("spark.serializer", SPARK_SERIALIZER) \



"""def modify_parameters(spark_session, file_path, dev_addr_list, params, target_values, dataset_format):
    start_time = time.time()

    # Load the dataset
    if dataset_format == "json":
        df = spark_session.read.json(file_path)
    else:
        df = spark_session.read.parquet(file_path)

    modified_parts = []
    cleaned_parts = []

    for dev_addr in dev_addr_list:
        df_dev = df.filter(col("DevAddr") == dev_addr)

        count_dev = df_dev.count()
        if count_dev == 0:
            continue

        num_intrusions = max(1, round(count_dev * 0.25))  # Ensure at least 1

        df_to_modify_base = df_dev.limit(num_intrusions)

        df_to_modify = df_to_modify_base.withColumn("unique_id", monotonically_increasing_id())
        window = Window.orderBy("unique_id")
        indexed = df_to_modify.withColumn("row_number", row_number().over(window) - 1)

        for param, values in zip(params, target_values):
            value_map = create_map([lit(i) for i in sum(zip(range(num_intrusions), values), ())])
            indexed = indexed.withColumn(param, value_map.getItem(col("row_number")))

        indexed = indexed.withColumn("intrusion", lit(1))

        # Anti-join to remove modified rows
        df_alias = df_dev.alias("df")
        mod_alias = df_to_modify_base.alias("mod")
        join_condition = [col(f"df.{c}") == col(f"mod.{c}") for c in df_to_modify_base.columns]
        df_cleaned = df_alias.join(mod_alias, on=join_condition, how="anti")

        modified_parts.append(indexed.drop("row_number", "unique_id"))
        cleaned_parts.append(df_cleaned)

    # Include rows not related to any DevAddr in the list
    untouched_df = df.filter(~col("DevAddr").isin(dev_addr_list))

    df_final = untouched_df.unionByName(spark_session.createDataFrame([], df.schema))  # empty placeholder
    if cleaned_parts:
        df_final = df_final.unionByName(cleaned_parts[0])
        for part in cleaned_parts[1:]:
            df_final = df_final.unionByName(part)

    for mod in modified_parts:
        df_final = df_final.unionByName(mod)

    # Save result
    if dataset_format == "json":
        df_final.coalesce(1).write.mode("overwrite").json(file_path)
    else:
        df_final.coalesce(1).write.mode("overwrite").parquet(file_path)

    end_time = time.time()
    print(f"File {file_path} successfully modified in {format_time(end_time - start_time)}")
"""


def modify_device_dataset(df, output_file_path, params, target_values, datasets_format):

    num_intrusions = round(df.count() * 0.25)
    
    # Select the first N logs directly with no sorting
    df_to_modify = df.limit(num_intrusions)

    # Add an index to map the intrusion values
    indexed = df_to_modify.rdd.zipWithIndex().toDF()
    indexed = indexed.selectExpr("_1.*", "_2 as row_number")

    # Apply intrusion values based on index
    for param, values in zip(params, target_values):
        value_map = create_map([lit(i) for i in sum(zip(range(num_intrusions), values), ())])
        indexed = indexed.withColumn(param, value_map.getItem(col("row_number")))

    # Mark packets as intrusive
    indexed = indexed.withColumn("intrusion", lit(1))

    # Prepare the non-modified dataframe
    df_unmodified = df.exceptAll(df_to_modify)

    # Join intrusive packets with normal packets
    df_final = df_unmodified.unionByName(indexed.drop("row_number"))

    # Save final dataframe in JSON or PARQUET format
    if datasets_format == "json":
        df_final.coalesce(1).write.mode("overwrite").json(output_file_path)
    else:
        df_final.coalesce(1).write.mode("overwrite").parquet(output_file_path)

    # Print the dataframe
    df_final.groupBy("intrusion").count().show()

    return df_final


def modify_dataset(df, output_file_path, params, target_values, dev_addr_list, datasets_format):

    modified_dfs = []

    # Loop for all selected devices 
    for dev_addr in dev_addr_list:

        # Filters dataframe to contain only packets from the device 'dev_addr'
        df_device = df.filter(col("DevAddr") == dev_addr)

        num_packets = df_device.count()

        if num_packets == 0:
            continue

        num_intrusions = round(num_packets * 0.25)

        # Select the first N logs directly with no sorting
        df_to_modify = df_device.limit(num_intrusions)

        # Add an index to map the intrusion values
        indexed = df_to_modify.rdd.zipWithIndex().toDF()
        indexed = indexed.selectExpr("_1.*", "_2 as row_number")

        # Apply intrusion values based on index
        for param, values in zip(params, target_values):
            mapping_expr = when(col("row_number") == 0, values[0])
            for i in range(1, len(values)):
                mapping_expr = mapping_expr.when(col("row_number") == i, values[i])
            indexed = indexed.withColumn(param, mapping_expr)

        # Mark modified packets as intrusive
        indexed = indexed.withColumn("intrusion", lit(1))

        # Prepare the non-modified dataframe
        df_unmodified = df_device.exceptAll(df_to_modify)

        # Join intrusive packets with normal packets
        df_device_final = df_unmodified.unionByName(indexed.drop("row_number"))

        modified_dfs.append(df_device_final)

    # Joins modified packets from all selected devices
    df_modified  = modified_dfs[0]
    for d in modified_dfs[1:]:
        df_modified = df_modified.unionByName(d)

    # Unmodified packets
    df_unmodified_devices = df.filter(~col("dev_addr").isin(dev_addr_list))

    # Final result
    df_final = df_modified.unionByName(df_unmodified_devices)

    # Save final dataframe in JSON or PARQUET format
    if datasets_format == "json":
        df_final.coalesce(1).write.mode("overwrite").json(output_file_path)
    else:
        df_final.coalesce(1).write.mode("overwrite").parquet(output_file_path)

    return df_final





"""## This function modifies the generated test dataset with introduced attacks, manipulating some parameters such as RSSI and LSNR
def modify_parameters(spark_session, file_path, dev_addr_list, params, target_values, dataset_format):
    start_time = time.time()

    # Load the full DataFrame
    if dataset_format == "json":
        df = spark_session.read.json(file_path)
    else:
        df = spark_session.read.parquet(file_path)

    # Filter DevAddrs of interest
    df_filtered = df.filter(col("DevAddr").isin(dev_addr_list))

    # Add row numbers per DevAddr partition
    window_spec = Window.partitionBy("DevAddr").orderBy(monotonically_increasing_id())
    df_with_rownum = df_filtered.withColumn("row_number", row_number().over(window_spec))

    # Number of rows to modify (25% per device)
    num_samples_dev_addr = round(df_with_rownum.count() * 0.2) * len(dev_addr_list)
    df_to_modify = df_with_rownum.filter(col("row_number") <= num_samples_dev_addr)

    # Add global row index for mapping values
    window_global = Window.orderBy(monotonically_increasing_id())
    df_to_modify = df_to_modify.withColumn("row_index", row_number().over(window_global) - 1)

    num_intrusions = df_to_modify.count()

    # Apply modifications per parameter
    for param in params:
        # Ensure scalar values
        if any(isinstance(x, list) for x in target_values):
            target_values = [x[0] if isinstance(x, list) else x for x in target_values]
        
        #value_map = create_map([lit(x) for pair in zip(range(num_intrusions), target_values[:num_intrusions]) for x in pair])
        df_to_modify = df_to_modify.withColumn(
            "intrusion",
            when(col("row_index") < num_intrusions, lit(1)).otherwise(lit(0))
        )

    # Drop helper columns
    df_to_modify = df_to_modify.drop("row_number", "row_index")

    # Select unmodified rows
    df_unmodified = df.join(df_to_modify.select("tmst"), on="tmst", how="left_anti")

    # Align schema types and column order before union
    for column in df.columns:
        if column not in df_to_modify.columns:
            df_to_modify = df_to_modify.withColumn(column, lit(None).cast(df.schema[column].dataType))
        else:
            df_to_modify = df_to_modify.withColumn(column, df_to_modify[column].cast(df.schema[column].dataType))

        if column not in df_unmodified.columns:
            df_unmodified = df_unmodified.withColumn(column, lit(None).cast(df.schema[column].dataType))
        else:
            df_unmodified = df_unmodified.withColumn(column, df_unmodified[column].cast(df.schema[column].dataType))

    # Reorder columns to match
    df_to_modify = df_to_modify.select(df.columns)
    df_unmodified = df_unmodified.select(df.columns)

    # Merge DataFrames
    df_final = df_unmodified.unionByName(df_to_modify)

    # Write modified DataFrame
    if dataset_format == "json":
        df_final.coalesce(1).write.mode("overwrite").json(file_path)
    else:
        df_final.coalesce(1).write.mode("overwrite").parquet(file_path)

    end_time = time.time()
    print(f"File {file_path} successfully modified in {format_time(end_time - start_time)}")
"""


""" TODO review this
Waits until Spark writes a valid JSON file inside the 'path' directory

"""
def wait_for_spark_json(path, timeout=30):
   
    start_time = time.time()
    
    while True:
        
        json_files = glob.glob(os.path.join(path, "part-*.json"))
       
        if json_files:
            return json_files[0]  # devolve o primeiro ficheiro encontrado
        
        if time.time() - start_time > timeout:
            raise FileNotFoundError(f"Timeout: Nenhum ficheiro JSON encontrado em '{path}' após {timeout} segundos.")
        
        time.sleep(1)  # espera 1s antes de tentar novamente


"""
This function ensures that there are always sufficient samples for both training and testing
considering the total number of examples in the dataframe corresponding to the device, if
the total number of samples is larger than 1

"""
def train_test_split(df_model, seed=42):

    # Count the total number of samples to be used by the model
    total_count = df_model.count()

    # If there is only one sample for the device, use that sample for training, 
    # and don't apply testing for that model
    if total_count == 1:
        df_model_train, df_model_test = df_model, None

    # If there are between 2 and 9 samples, split the samples for training and testing by 50-50
    elif total_count < 10:
        df_model_train, df_model_test = df_model.randomSplit([0.5, 0.5], seed)

    # If there are between 10 and 20 samples, split the samples for training and testing by 70-30
    elif total_count < 20:
        df_model_train, df_model_test = df_model.randomSplit([0.7, 0.3], seed)

    # If there are 20 or more samples, split the samples for training and testing by 80-20
    else:
        df_model_train, df_model_test = df_model.randomSplit([0.8, 0.2], seed)

    return df_model_train, df_model_test



"""
Auxiliary function to get all attributes names of a spark dataframe schema

    schema - spark dataframe schema
    parent_name - name of parent field of an array or struct. Optional, only applicable of the field is array or struct and
                    used for recursive calls inside the function.

Returns: a array

"""
def get_all_attributes_names(df_schema, parent_name=""):
    
    attribute_names = []

    # Iterate through all the fields in the schema, including fields inside arrays and structs
    for field in df_schema.fields:

        if isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
            attribute_names.extend(get_all_attributes_names(field.dataType.elementType, field.name))  # Recursive call for nested structs
        
        elif isinstance(field.dataType, StructType):
            attribute_names.extend(get_all_attributes_names(field.dataType, field.name))  # Handle direct nested structs
    
        else:
            attribute_names.append(field.name)

    return attribute_names


"""
Auxiliary function to get all boolean attributes names of a spark dataframe schema

    schema - spark dataframe schema
    parent_name - name of parent field of an array or struct. Optional, only applicable of the field is array or struct and
                    used for recursive calls inside the function.

Returns: a array

"""
def get_boolean_attributes_names(df_schema, parent_name=""):
    boolean_names = []

    for field in df_schema.fields:
        full_name = f"{parent_name}.{field.name}" if parent_name else field.name

        if isinstance(field.dataType, ArrayType):
            if isinstance(field.dataType.elementType, StructType):
                # Array of structs: recursive call
                boolean_names.extend(get_boolean_attributes_names(field.dataType.elementType, full_name))
        
        elif isinstance(field.dataType, StructType):
            # Nested struct: recursive call
            boolean_names.extend(get_boolean_attributes_names(field.dataType, full_name))

        elif isinstance(field.dataType, BooleanType):
            # It's a boolean field: add full path name
            boolean_names.append(full_name)
    
    return boolean_names
