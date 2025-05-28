
from common.constants import *
from pyspark.sql import SparkSession
from pyspark.sql.types import *

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
                            .getOrCreate()
                            #.config("spark.serializer", SPARK_SERIALIZER) \



"""
This function ensures that there are always sufficient samples for both training and testing
considering the total number of examples in the dataframe corresponding to the device, if
the total number of samples is larger than 1

"""
def train_test_split(df_model, seed=422):

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

    # If there are 20 or more samples, split the samples for training and testing by 85-15
    else:
        df_model_train, df_model_test = df_model.randomSplit([0.85, 0.15], seed)

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
