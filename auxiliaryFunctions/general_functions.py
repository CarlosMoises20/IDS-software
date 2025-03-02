
import os
from pyspark.sql.types import *


# Auxiliary function to bind all log files inside a indicated directory
# into a single log file of each type of LoRaWAN message
def bind_dir_files(dataset_root_path, dataset_type):

    output_filename = f"./combinedDatasets/combined_logs_{dataset_type.value}.log"

    # Skip file generation if it already exists
    if os.path.exists(output_filename):
        print(f"File '{output_filename}' already exists. Skipping generation.")
    
    else:
        all_logs = []           # Create a list to store the content of different files

        dataset_from_type = [os.path.join(os.fsdecode(dataset_root_path), os.fsdecode(file))
                    for file in os.listdir(dataset_root_path) if file.decode().startswith(dataset_type.value)]

        # Loop through all files in the directory
        for filename in dataset_from_type:
            with open(filename, 'r') as f:
                all_logs.append(f.read())     # Append the contents of the file to the list

        # Join all logs into a single string
        combined_logs = '\n'.join(all_logs)

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Write the combined logs to a new file
        with open(output_filename, 'w') as f:
            f.write(combined_logs)


    return output_filename


"""
Function that converts a hexadecimal number inside a string to a decimal number as an integer 

"""
def hex_to_decimal(hex_string):
    
    # Check if string is Null
    if hex_string == None:
        return None
    
    # Convert the hexadecimal string to decimal (base 10)
    return int(hex_string, 16)


"""
Function that converts a hexadecimal number inside a string to a binary number as an integer 

"""
def hex_to_binary(hex_string):
    
    # Check if string is Null
    if hex_string == None:
        return None
    
    # Convert the hexadecimal string to binary (base 2)
    return int(bin(int(hex_string, 16))[2:])


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