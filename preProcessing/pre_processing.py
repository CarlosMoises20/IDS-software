
import re
from pyspark.sql.types import StringType
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
    def hex_to_dec(hex_str):
        try:
            return str(int(hex_str, 16))  # Convert hex to decimal as string
        except ValueError:
            return "-1"  # Default value for invalid hex


    """
    Method to convert hexadecimal attributes to decimal attributes in numeric format (DecimalType(38, 0)),
    in order to avoid loss of precision and accuracy in very big values, supporting a scale of up to 38 digits
    
        df: spark dataframe that represents the dataset
        attributes: a list of strings with the names of attributes to be converted
    
    """
    @staticmethod
    def hex_to_decimal(df, attributes):

        # Define UDF with StringType() to support any size
        hex_to_dec_udf = udf(DataPreProcessing.hex_to_dec, StringType())
        
        for attr in attributes: 

            # Fill missing values (None or empty strings) with -1, since -1 would never be a valid value
            # for an hexadecimal-to-decimal attribute
            df = df.withColumn(attr, when((col(attr).isNull()) | (col(attr) == lit("")), lit(-1))
                                      .otherwise(hex_to_dec_udf(col(attr))))

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

