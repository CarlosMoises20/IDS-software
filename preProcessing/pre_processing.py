
import re
from pyspark.sql.types import StringType, IntegerType, DecimalType
from abc import ABC, abstractmethod
from pyspark.sql.functions import when, col, expr, lit, udf
from auxiliaryFunctions.general import get_all_attributes_names
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from decimal import Decimal, InvalidOperation

class DataPreProcessing(ABC):


    # TODO: review this (avoid VectorAssembler if possible, or if that is not possible, find a way to normalize
    # attributes and return the result in the attributes themselves and remove the columns used exclusively VectorAssembler)
    @staticmethod
    def normalization(df):

        # Normalize all attributes except DevAddr that will not be used for model training, only to identify the model
        attributes = list(set(get_all_attributes_names(df.schema)) - set(["DevAddr"]))

        assembler = VectorAssembler(inputCols=attributes, outputCol="features")

        df = assembler.transform(df)
        
        scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
        df = scaler.fit(df).transform(df)
        
        df = df.drop("features")

        return df
    
    @staticmethod
    def bool_to_int(df, attributes):

        for attr in attributes: 

            # Fill missing values (None or empty strings) with -1, since -1 would never be a valid value
            # for a boolean-to-integer attribute; and also for ML algorithms to be capable to process better the
            # values, since NULL values can contribute to poor performance of these algorithms
            df = df.withColumn(attr, when(col(attr).isNull(), -1)
                                     .otherwise(col(attr).cast(IntegerType())))

        return df
    
    @staticmethod
    def str_to_float(fraction_str):
        try:
            
            # Split the string at the slash and convert the parts to integers
            numerator, denominator = fraction_str.split("/")
            
            # Convert parts from string to integers
            numerator = int(numerator)
            denominator = int(denominator)

            # Perform the division to get the result in float
            return numerator / denominator
        
        except:
            # Return -1 if there was an error
            return -1

    """
    Method to convert hexadecimal attributes to decimal attributes in numeric format (DecimalType(38, 0)),
    in order to avoid loss of precision and accuracy in very big values, supporting a scale of up to 38 digits
    
        df: spark dataframe that represents the dataset
        attributes: a list of strings with the names of attributes to be converted
    
    """
    @staticmethod
    def hex_to_decimal(df, attributes):
        
        for attr in attributes: 

            # Fill missing values (None or empty strings) with -1, since -1 would never be a valid value
            # for an hexadecimal-to-decimal attribute
            df = df.withColumn(attr, when((col(attr).isNull()) | (col(attr) == lit("")), lit(-1))
                                      .otherwise(expr(f"conv({attr}, 16, 10)").cast(DecimalType(38, 0))))

        return df

    """
    Method to reverse hexadecimal octets in string format
    
        hex_str: hexadecimal value
    
    """
    @staticmethod
    def reverse_hex_octets(hex_str):

        # If hex_str is None or an empty string, return an empty string
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

