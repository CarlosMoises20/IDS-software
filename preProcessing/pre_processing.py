
from pyspark.sql.types import *
from abc import ABC, abstractmethod
from pyspark.sql.functions import when, col, expr


class DataPreProcessing(ABC):

    """
    Method to reverse hexadecimal octets in string format
    
        hex_str: hexadecimal value
    
    """
    @staticmethod
    def reverse_hex_octets(hex_str):

        # If hex_str is None, return None
        if hex_str is None:
            return None

        # Ensure hex_str has an even number of characters
        if len(hex_str) % 2 != 0:
            raise ValueError("Invalid Format")
        
        # Divides hex_str into octets
        octets = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
        reversed_octets = "".join(reversed(octets))
        
        return reversed_octets



    # TODO: fix
    @staticmethod
    def str_to_float(df, attributes):

        for attr in attributes:
            df = df.withColumn(attr, when(col(attr).isNull(), None)
                                      .otherwise(col(attr).cast("float")))

        return df

    """
    Method to convert hexadecimal attributes to decimal attributes in integer format
    
        df: spark dataframe that represents the dataset
        attributes: a list of strings with the names of attributes to be converted
    
    """
    @staticmethod
    def hex_to_decimal(df, attributes):
        
        for attr in attributes:
            df = df.withColumn(attr, when(col(attr).isNull(), None)
                                      .otherwise(expr(f"conv({attr}, 16, 10)").cast(IntegerType())))

        return df


    """
    Method to convert hexadecimal attributes to decimal attributes in string format
    
        df: spark dataframe that represents the dataset
        attributes: a list of strings with the names of attributes to be converted
    
    """
    @staticmethod
    def hex_to_binary(df, attributes):
        
        for attr in attributes:
            df = df.withColumn(attr, when(col(attr).isNull(), None)
                                      .otherwise(expr(f"bin(conv({attr}, 16, 10))").cast(IntegerType())))

        return df

    
    """
    Abstract method for data pre-processing that has different implementations for 'rxpk' and 'txpk
    
    It's used to apply feature selection techniques to remove irrelevant, redundant and correlated attributes,
    and also to apply necessary transformations on dataset relevant attributes

        df: spark dataframe that represents the dataset

        return: pre-processed spark dataframe 

    """
    @staticmethod
    @abstractmethod
    def pre_process_data(df):
        """..."""

