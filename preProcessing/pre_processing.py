
from pyspark.sql.types import *
from abc import ABC, abstractmethod
from pyspark.sql.functions import when


class DataPreProcessing(ABC):

    """
    Method to convert boolean attributes to integer attributes
    
        df: spark dataframe that represents the dataset
        attributes: a list of strings with the names of attributes to be converted
    
    """
    @staticmethod
    def bool_to_int(df, attributes):
        
        for attr in attributes:
            df = df.withColumn(attr, when(df[attr] == True, 1)
                                      .when(df[attr] == False, 0)
                                      .otherwise(None)) 
            
        return df


    
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
        reversed_octets = "".join(reversed([hex_str[i:i+2] for i in range(0, len(hex_str), 2)]))
        
        return reversed_octets

    
    """
    Method to convert hexadecimal attributes to decimal attributes in integer format
    
        df: spark dataframe that represents the dataset
        attributes: a list of strings with the names of attributes to be converted
    
    """
    @staticmethod
    def hex_to_decimal(df, attributes):
        
        for attr in attributes:
            df = df.withColumn(attr, when(df[attr] == None, None)
                                      .otherwise(int(df[attr], 16)))


        return df


    """
    Method to convert hexadecimal attributes to decimal attributes in string format
    
        df: spark dataframe that represents the dataset
        attributes: a list of strings with the names of attributes to be converted
    
    """
    @staticmethod
    def hex_to_binary(df, attributes):
        
        for attr in attributes:
            df = df.withColumn(attr, when(df[attr] == None, None)
                                      .otherwise(str(bin(int(df[attr], 16))[2:])))

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

