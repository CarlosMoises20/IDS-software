
import os
from pyspark.sql.types import *


"""
Auxiliary function to aggregate all log files inside a directory,
into a single file corresponding to each type of LoRaWAN message

    dataset_root_path (string) - path to the directory containing the log files
    dataset_type (DatasetType enum) - type of dataset to be processed (DatasetType.RXPK or DatasetType.TXPK in this case)

It returns the name of the output file to be used to load dataset as a spark dataframe

"""
def bind_dir_files(spark_session, dataset_type):

    dataset_root_path = os.fsencode("./datasets")

    # Define the name of the file where all logs corresponding 
    # to the dataset type 'dataset_type' will be stored
    output_filename = f'./combinedDatasets/combined_logs_{dataset_type.value["name"]}.log'

    # Skip file generation if it already exists
    if os.path.exists(output_filename):
        print(f"File '{output_filename}' already exists. Skipping generation.")
    
    # Generate file if it doesn't exist yet
    else:
        # Create a list to store the content of different files
        all_logs = []           

        # Filter files based on the dataset type
        filenames_from_type = [os.path.join(os.fsdecode(dataset_root_path), os.fsdecode(file))
                    for file in os.listdir(dataset_root_path) if file.decode().startswith(dataset_type.value["name"])]

        # Loop through all files in the directory
        for filename in filenames_from_type:
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

        print(f"File '{output_filename}' created")


    # Define the output Parquet filename
    parquet_filename = output_filename.replace(".log", ".parquet")

    if os.path.exists(parquet_filename):
        print(f"Parquet file {parquet_filename} already exists. Skipping generation")

    else:
        # Read the .log file as a text DataFrame
        df = spark_session.read.json(output_filename)

        # Write DataFrame to Parquet
        df.write.parquet(parquet_filename)

        print(f"File '{parquet_filename}' created")

    # Returns the name of the output parquet file to be used to load dataset as a spark dataframe
    # This is done since processing in spark using parquet files is faster than if log files with JSON format were used 
    return parquet_filename


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


"""
Auxiliary function to print processing time on an adequate format

"""
def format_time(seconds):

    # Convert seconds to milisseconds (round to integer)
    milisseconds = round(seconds * 1000)

    # If the time is less than 1 second and arrendondated value of milisseconds is less than 1000 milisseconds
    # For example: seconds = 0.9999 -> milisseconds = 999.9 =~ 1000 and time will be printed in seconds
    if milisseconds < 1000:
        return f"{milisseconds} ms"      # Milisseconds

    else:
        # Function to round the value of seconds to integer
        seconds = round(seconds)
    
        if seconds < 60:
            return f"{seconds:.2f} s "                      # Seconds
        
        elif seconds < 3600:
            minutes = seconds // 60                         # Minute as the integer part of the division of total by number of seconds in a minute
            secs = seconds % 60                             # Seconds as the rest of the division of total by number of seconds in a minute
            return f"{minutes} min {secs} s "               # Minutes and seconds
        
        else:
            hours = seconds // 3600                         # Hour as the integer part of the division of total by number of seconds in a hour
            minutes = (seconds % 3600) // 60                # Minutes as the rest of the division of total by number of seconds in a hour and integer part of division by number of seconds in a minute
            secs = seconds % 60                             # Seconds as the rest of the division of total by number of seconds in a minute
            return f"{hours} h {minutes} min {secs} s "     # Minutes, hours and seconds

