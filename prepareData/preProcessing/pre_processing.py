
from pyspark.sql.types import IntegerType, DecimalType
from abc import ABC, abstractmethod
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.mllib.linalg import Vectors as MLLibVectors
from decimal import Decimal
from pyspark.sql import functions as F
from common.spark_functions import get_all_attributes_names
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler, PCA
from pyspark.ml.linalg import DenseVector as MLDenseVector


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
    def features_assembler(df_train, df_test, explained_variance_threshold=0.99):

        # Asseble all attributes except DevAddr, intrusion and prediction that will not be used for model training, only to identify the model
        column_names = list(set(get_all_attributes_names(df_train.schema)) - set(["DevAddr", "intrusion", "prediction", "score"]))
        assembler = VectorAssembler(inputCols=column_names, outputCol="scaled")
        df_train = assembler.transform(df_train)
        df_test = assembler.transform(df_test)

        """# Normalize all assembled features inside a scale
        scaler = MinMaxScaler(inputCol="feat", outputCol="features", max=100000)
        scaler_model = scaler.fit(df_train)
        df_train = scaler_model.transform(df_train)
        df_test = scaler_model.transform(df_test)"""

        """# Normalize all assembled features using standards
        # TODO test, for all algorithms, with mean, with std, without mean, without std
        scaler = StandardScaler(inputCol="feat", outputCol="scaled", withMean=True, withStd=True)
        scaler_model = scaler.fit(df_train)

        df_train = scaler_model.transform(df_train)
        df_test = scaler_model.transform(df_test)"""
        
        """### PCA (Principal Component Analysis)
            
        # Fit PCA using the train dataset    
        pca = PCA(k=len(column_names), inputCol="scaled", outputCol="features")
        pca_model = pca.fit(df_train)
        explained_variance = pca_model.explainedVariance.cumsum()

        # Determine the optimal k, that allows to capture at least 'explained_variance_threshold'*100 % of the variance
        k_optimal = next(i + 1 for i, v in enumerate(explained_variance) if v >= explained_variance_threshold)

        # Do the same thing but with the determined optimal k (k_optimal)
        pca = PCA(k=k_optimal, inputCol="scaled", outputCol="features")
        pca_final_model = pca.fit(df_train)

        # Applies trained PCA model to train and test dataset
        df_train = pca_final_model.transform(df_train)
        df_test = pca_final_model.transform(df_test)

        # Prints the chosen value for k
        print(f"Optimal number of PCA components: {k_optimal} (explaining {explained_variance[k_optimal-1]*100:.2f}% of the variance)")"""

        ### SVD: Converts for appropriate format for RowMatrix
        rdd_vectors = df_train.select("scaled").rdd.map(lambda row: MLLibVectors.dense(row["scaled"]))
        mat = RowMatrix(rdd_vectors)

        # Applies SVD (maximum k = número de colunas)
        k_max = len(column_names)
        svd = mat.computeSVD(k_max, computeU=False)

        # Calculates cumulated explained variance (according to the singular values)
        sigma = svd.s.toArray()
        total_variance = (sigma ** 2).sum()
        explained_variance = (sigma ** 2).cumsum() / total_variance

        # Determines the ideal k
        k_optimal = next(i + 1 for i, v in enumerate(explained_variance) if v >= explained_variance_threshold)

        print(f"Optimal number of SVD components: {k_optimal} (explaining {explained_variance[k_optimal-1]*100:.2f}% of the variance)")

        # Projects data for the k principal components through manual multiplying (using only top-k V)
        V = svd.V.toArray()[:, :k_optimal]

        # Manually applies the transformation feat * V to obtain the new reduced vectors
        def project_features(feat):
            # feat is a DenseVector from pyspark.ml.linalg, so feat.dot(V) works
            projected_array = feat.dot(V)
            return MLDenseVector(projected_array)

        project_udf = F.udf(project_features, returnType=VectorUDT())
        df_train = df_train.withColumn("features", project_udf("scaled"))
        df_test = df_test.withColumn("features", project_udf("scaled"))

        return df_train.drop("feat", "scaled"), df_test.drop("feat", "scaled")
       
    """
    Method to convert boolean attributes to integer attributes in numeric format (IntegerType())
    
        df: spark dataframe that represents the dataset
        attributes: list of attributes to convert to integer
        
        return: spark dataframe with converted attributes
    
    """
    @staticmethod
    def bool_to_int(df, attributes):

        for attr in attributes: 

            df = df.withColumn(attr, F.when(F.col(attr).isNull(), None)
                                     .otherwise(F.col(attr).cast(IntegerType())))

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
        # NULL or EMPTY STRING: NULL
        # ANOMALY: -1
        def hex_to_decimal_int(hex_str):
    
            if hex_str is None or hex_str == "":
                return None
            
            try:
                res = int(hex_str, 16)
                
                # When the size is higher than expected, indicate an anomaly through another negative number
                if len(str(res)) > 38:
                    return Decimal(-1)

                return Decimal(int(hex_str, 16))
            
            except:
                return Decimal(-1)
                
        hex_to_decimal_udf = F.udf(hex_to_decimal_int, DecimalType(38, 0))

        for attr in attributes: 

            # Fill missing values (None or empty strings) with -1, since -1 would never be a valid value
            # for an hexadecimal-to-decimal attribute
            df = df.withColumn(attr, hex_to_decimal_udf(F.col(attr)))

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

