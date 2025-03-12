
from pyspark.sql.types import DecimalType
from abc import ABC, abstractmethod
from pyspark.sql.functions import when, col, expr, lit, bool_or

class DataPreProcessing(ABC):

    """
    Method to reverse hexadecimal octets in string format
    
        hex_str: hexadecimal value
    
    """
    @staticmethod
    def reverse_hex_octets(hex_str):

        # If hex_str is None, return None
        if (hex_str is None) or (hex_str == ""):
            return -1

        # Ensure hex_str has an even number of characters
        if len(hex_str) % 2 != 0:
            raise ValueError("Invalid Format")
        
        # Divides hex_str into octets
        octets = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
        reversed_octets = "".join(reversed(octets))
        
        return reversed_octets


    """
    Method to convert hexadecimal attributes to decimal attributes in numeric format (DecimalType(38, 0)),
    in order to avoid loss of precision and accuracy in very big values, supporting a scale of up to 38 digits
    
        df: spark dataframe that represents the dataset
        attributes: a list of strings with the names of attributes to be converted
    
    """
    @staticmethod
    def hex_to_decimal(df, attributes):
        
        for attr in attributes:
            # Fill missing values with -1, since -1 would never be a valid value
            # for an hexadecimal-to-decimal attribute
            df = df.withColumn(attr, when((col(attr).isNull()) | (col(attr) == lit("")), -1)
                                      .otherwise(expr(f"conv({attr}, 16, 10)").cast(DecimalType(38, 0))))

        return df


    """
    Method to convert hexadecimal attributes to decimal attributes in string format
    
        df: spark dataframe that represents the dataset
        attributes: a list of strings with the names of attributes to be converted
    
    """
    @staticmethod
    def hex_to_binary(df, attributes):
        
        for attr in attributes:
            df = df.withColumn(attr, when((col(attr).isNull()) | (col(attr) == lit("")), -1)
                                      .otherwise(expr(f"bin(conv({attr}, 16, 10))")))

        return df

    
    """
    Abstract method for data pre-processing that has different implementations for 'rxpk' and 'txpk'
    
    It's used to apply feature selection techniques to remove irrelevant, redundant and correlated attributes,
    and also to apply necessary transformations on dataset relevant attributes

        df: spark dataframe that represents the dataset

        return: pre-processed spark dataframe 

    """
    @staticmethod
    @abstractmethod
    def pre_process_data(df):
        """..."""

