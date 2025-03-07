
from preProcessing.pre_processing import *
from auxiliaryFunctions.general import *
from auxiliaryFunctions.anomaly_detection import *
from abc import ABC, abstractmethod


class DataProcessing(ABC):


    
    """
    Auxiliary function to get all attributes names of a spark dataframe schema

        schema - spark dataframe schema
        parent_name - name of parent field of an array or struct. Optional, only applicable of the field is array or struct and
                        used for recursive calls inside the function.

    Returns: a array

    """
    @staticmethod
    def get_all_attributes_names(df_schema, parent_name=""):
        
        attribute_names = []

        # Iterate through all the fields in the schema, including fields inside arrays and structs
        for field in df_schema.fields:

            if isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
                attribute_names.extend(DataProcessing.get_all_attributes_names(field.dataType.elementType, field.name))  # Recursive call for nested structs
            
            elif isinstance(field.dataType, StructType):
                attribute_names.extend(DataProcessing.get_all_attributes_names(field.dataType, field.name))  # Handle direct nested structs
        
            else:
                attribute_names.append(field.name)

        return attribute_names


    @staticmethod
    @abstractmethod
    def process_data(df_train, df_test):
        """..."""