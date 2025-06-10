
from pyspark.sql.types import IntegerType, DecimalType
from abc import ABC, abstractmethod
from pyspark.sql.functions import when, col, expr, lit, udf
from decimal import Decimal
from common.spark_functions import get_all_attributes_names
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, PCA


class DataPreProcessing(ABC):

    """
    Applies several techniques and assembles to features.
    
    This improves the performance of machine learning algorithms by enabling faster convergence, better clustering,
    and more consistent input for neural networks.

    Without normalization, features with wider value ranges can dominate those with smaller ranges, 
    due to their higher variance. By normalizing all features to the same scale, each one contributes equally 
    to the learning process.

    Note: The 'DevAddr' column is excluded from normalization, as it is only used to identify the model,
    not for training.

    """
    @staticmethod
    def features_assembler(df, explained_variance_threshold=0.95):

        # Asseble all attributes except DevAddr, intrusion and prediction that will not be used for model training, only to identify the model
        column_names = list(set(get_all_attributes_names(df.schema)) - set(["DevAddr", "intrusion", "prediction", "score"]))

        assembler = VectorAssembler(inputCols=column_names, outputCol="feat")

        df = assembler.transform(df)
        
        # Normalize all assembled attributes
        #scaler = MinMaxScaler(inputCol="feat", outputCol="features")
        
        #df = scaler.fit(df).transform(df)

        #return df.drop("feat")

        # Applies PCA for dimensionality reduction
        pca = PCA(k=len(column_names), inputCol="feat", outputCol="features")
        model = pca.fit(df)

        # Soma acumulada da variância explicada
        explained_variance = model.explainedVariance.cumsum()

        # Define o menor k que atinge o limiar de variância desejada
        k_optimal = next(i + 1 for i, v in enumerate(explained_variance) if v >= explained_variance_threshold)

        # Aplica novamente o PCA com o número ótimo de componentes
        pca_final = PCA(k=k_optimal, inputCol="feat", outputCol="features")
        df = pca_final.fit(df).transform(df)

        # Mostra o valor escolhido para k (pode ser removido em produção)
        print(f"Número ótimo de componentes PCA: {k_optimal} (explicando {explained_variance[k_optimal-1]*100:.2f}% da variância)")

        return df
       
    """
    Method to convert boolean attributes to integer attributes in numeric format (IntegerType())
    
        df: spark dataframe that represents the dataset
        attributes: list of attributes to convert to integer
        
        return: spark dataframe with converted attributes
    
    """
    @staticmethod
    def bool_to_int(df, attributes):

        for attr in attributes: 

            # Fill missing values (None or empty strings) with -1, since -1 would never be a valid value
            # for a boolean-to-integer attribute; and also for ML algorithms to be capable to process better the
            # values, since NULL values can contribute to poor performance of these algorithms
            df = df.withColumn(attr, when(col(attr).isNull(), -1)
                                     .otherwise(col(attr).cast(IntegerType())))

        return df
    
    """
    Convert "datr" attribute from string to float
    
        Example = "4/5" -> 0.8
    
        fraction_str: string that represents a fraction
        
        return: float
        
    """
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

        # NOT NULL: Converted number to decimal
        # NULL or EMPTY STRING: -1
        # ANOMALY: -2
        def hex_to_decimal_int(hex_str):
    
            if hex_str is None or hex_str == "":
                return Decimal(-1)
            
            try:
                res = int(hex_str, 16)
                
                # When the size is higher than expected, indicate an anomaly through another negative number
                if len(str(res)) > 38:
                    return Decimal(-2)

                return Decimal(int(hex_str, 16))
            
            except:
                return Decimal(-2)
                
        hex_to_decimal_udf = udf(hex_to_decimal_int, DecimalType(38, 0))

        for attr in attributes: 

            # Fill missing values (None or empty strings) with -1, since -1 would never be a valid value
            # for an hexadecimal-to-decimal attribute
            df = df.withColumn(attr, hex_to_decimal_udf(col(attr)))

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

