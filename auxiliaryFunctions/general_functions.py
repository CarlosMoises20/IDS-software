
import os
from pyspark.sql.types import *


"""
Auxiliary function to bind all log files inside an indicated directory
into a single log file of each type of LoRaWAN message

    dataset_root_path (string) - path to the directory containing the log files
    dataset_type (DatasetType enum) - type of dataset to be processed (DatasetType.RXPK or DatasetType.TXPK)

It returns the name of the output file to be used to load dataset as a spark dataframe

"""
def bind_dir_files(dataset_root_path, dataset_type):

    dataset_root_path = os.fsencode(dataset_root_path)

    # Define the name of the file where all logs corresponding 
    # to the dataset type 'dataset_type' will be stored
    output_filename = f'./combinedDatasets/combined_logs_{dataset_type.value["filename_field"]}.log'

    # Skip file generation if it already exists
    if os.path.exists(output_filename):
        print(f"File '{output_filename}' already exists. Skipping generation.")
    
    else:
        # Create a list to store the content of different files
        all_logs = []           

        # Filter files based on the dataset type
        dataset_from_type = [os.path.join(os.fsdecode(dataset_root_path), os.fsdecode(file))
                    for file in os.listdir(dataset_root_path) if file.decode().startswith(dataset_type.value["filename_field"])]

        # Loop through all files in the directory
        for filename in dataset_from_type:
            with open(filename, 'r') as f:
                all_logs.append(f.read())     # Append the contents of the file to the list

        # Join all logs into a single string
        combined_logs = '\n'.join(all_logs)

        # Ensure that the output directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Write the combined logs to a new file
        with open(output_filename, 'w') as f:
            f.write(combined_logs)

    # Returns the name of the output file to be used to load dataset as a spark dataframe
    return output_filename



"""
Auxiliary function to indicate if type of object 'obj' belongs to one of the 
instances inside the array 'instances'. If so, it returns True. Otherwise, it
returns False. 

"""
def is_one_of_instances(obj, instances):
    
    for instance in instances:
        if isinstance(obj, instance):
            return True
        
    return False


"""
Auxiliary function to get all numeric colums of a spark dataframe schema

    schema - spark dataframe schema
    parent_name - name of parent field of an array or struct. Optional, only applicable of the field is array or struct and
                    used for recursive calls inside the function.

Returns: a array

"""
def get_numeric_attributes(schema, parent_name=""):
    
    numeric_attributes = []

    # Iterate through all the fields in the schema, including fields inside arrays and structs
    for field in schema.fields:
        
        if is_one_of_instances(field.dataType, [DoubleType, FloatType, IntegerType, LongType, ShortType]):
            numeric_attributes.append(field.name)
        
        elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
            numeric_attributes.extend(get_numeric_attributes(field.dataType.elementType, field.name))  # Recursive call for nested structs
        
        elif isinstance(field.dataType, StructType):
            numeric_attributes.extend(get_numeric_attributes(field.dataType, field.name))  # Handle direct nested structs
    
    return numeric_attributes


"""
Auxiliary function to get all string colums of a spark dataframe schema

    schema - spark dataframe schema
    parent_name - name of parent field of an array or struct. Optional, only applicable of the field is array or struct and
                    used for recursive calls inside the function.

Returns: a array

"""
def get_string_attributes(df_schema, parent_name=""):
    
    string_attributes = []

    # Iterate through all the fields in the schema, including fields inside arrays and structs
    for field in df_schema.fields:
        
        if isinstance(field.dataType, StringType):
            string_attributes.append(field.name)
        
        elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
            string_attributes.extend(get_string_attributes(field.dataType.elementType, field.name))  # Recursive call for nested structs
        
        elif isinstance(field.dataType, StructType):
            string_attributes.extend(get_string_attributes(field.dataType, field.name))  # Handle direct nested structs
    
    return string_attributes


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