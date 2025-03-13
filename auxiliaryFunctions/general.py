
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
    
    # Generate file if it doesn't exist yet
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




# Auxiliary function to print time on adequate format
def format_time(seconds):
    
    # Function to round the value of seconds to integer
    seconds = round(seconds)

    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"            # Milisseconds
    elif seconds < 60:
        return f"{seconds:.2f} s "                   # Seconds
    elif seconds < 3600:
        minutes = int(seconds // 60)
        sec = int(seconds % 60)
        return f"{minutes} min {sec} s "             # Minutes and seconds
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        sec = int(seconds % 60)
        return f"{hours} h {minutes} min {sec} s "   # Minutes, hours and seconds