
from pyspark.sql.types import IntegerType, DecimalType
from abc import ABC, abstractmethod
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.linalg import VectorUDT
from pyspark.mllib.linalg import Vectors as MLLibVectors
from decimal import Decimal
from pyspark.sql import functions as F
from common.spark_functions import get_all_attributes_names
from pyspark.ml.feature import StandardScaler, VectorAssembler, PCA
from pyspark.ml.linalg import DenseVector as MLDenseVector
from common.model_type import ModelType
import base64

"""
This is an abstract class that includes methods used for generic pre-processing steps, namely assemble features in a
column that is used by machine learning, and manipulating attributes in a dataset (spark dataframe)
models for training and testing, and also an abstract method used by the classes that inherit from it, to apply pre-processing steps
that are only specific to a type of LoRaWAN messages

"""
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
    def features_assembler(df_train, df_test, model_type, explained_variance_threshold=0.99):

        # Asseble all attributes except DevAddr, intrusion and prediction that will not be used for model training, only to identify the model
        column_names = list(set(get_all_attributes_names(df_train.schema)) - set(["DevAddr", "intrusion"]))
        
        assembler = VectorAssembler(inputCols=column_names, outputCol="feat")
        df_train = assembler.transform(df_train)

        if df_test is not None:
            df_test = assembler.transform(df_test)

        """Normalize all assembled features using standards; with mean 0 and standard deviation 1;
        this allows all attributes' values to be centered on the mean 0 and have unit variance; this is
        important because several ML algorithms are sensitive to the scale of the numeric values and also to their variance
        and this scaling ensures that an attribute isn't considered by the model more important just because its variance or scale
        is higher or lower; instead, all features contribute equally to distance-based computations and model training;
        
        """
        scaler = StandardScaler(inputCol="feat", outputCol="scaled", withMean=True, withStd=True)
        scaler_model = scaler.fit(df_train)

        df_train = scaler_model.transform(df_train)
        
        if df_test is not None:
            df_test = scaler_model.transform(df_test)

        pca_final_model, svd_matrix = None, None
        
        if model_type in [ModelType.IF_CUSTOM, ModelType.LOF]:

            ### PCA (Principal Component Analysis)
                
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
            
            if df_test is not None:
                df_test = pca_final_model.transform(df_test)

            # Prints the chosen value for k
            print(f"Optimal number of PCA components: {k_optimal} (explaining {explained_variance[k_optimal-1]*100:.2f}% of the variance)")

        elif model_type in [ModelType.OCSVM, ModelType.HBOS, ModelType.KNN]:

            ### SVD (Singular Value Decomposition): Converts for appropriate format for RowMatrix
            rdd_vectors = df_train.select("scaled").rdd.map(lambda row: MLLibVectors.dense(row["scaled"]))
            mat = RowMatrix(rdd_vectors)

            # Applies SVD (maximum k = número de colunas)
            k_max = len(column_names)
            svd = mat.computeSVD(k_max)

            # Calculates cumulated explained variance (according to the singular values)
            sigma = svd.s.toArray()
            total_variance = (sigma ** 2).sum()
            explained_variance = (sigma ** 2).cumsum() / total_variance

            # Determines the ideal k; that is the minimum number of necessary SVD components to capture at least
            # 'explained_variance_threshold'*100 % of the variance of the dataset
            k_optimal = next(i + 1 for i, v in enumerate(explained_variance) if v >= explained_variance_threshold)

            print(f"Optimal number of SVD components: {k_optimal} (explaining {explained_variance[k_optimal-1]*100:.2f}% of the variance)")

            # Projects data for the k principal components through manual multiplying (using only top-k V)
            V = svd.V.toArray()[:, :k_optimal]

            # Manually applies the transformation feat * V to obtain the new reduced vectors
            def project_features(feat):
                # feat is a DenseVector from pyspark.ml.linalg, so feat.dot(V) works
                projected_array = feat.dot(V)
                return MLDenseVector(projected_array)

            svd_matrix = V

            project_udf = F.udf(project_features, returnType=VectorUDT())
            df_train = df_train.withColumn("features", project_udf("scaled"))
            
            if df_test is not None:
                df_test = df_test.withColumn("features", project_udf("scaled"))

        else:
            df_train = df_train.withColumnRenamed("scaled", "features")
            
            if df_test is not None:
                df_test = df_test.withColumnRenamed("scaled", "features")

        if df_test is not None:
            return df_train.drop("feat", "scaled"), df_test.drop("feat", "scaled"), {"StdScaler": scaler_model, 
                                                                                 "PCA": pca_final_model, 
                                                                                 "SVD": svd_matrix}
        
        return df_train.drop("feat", "scaled"), None, {"StdScaler": scaler_model, 
                                                        "PCA": pca_final_model, 
                                                        "SVD": svd_matrix}
    
    """
    This method applies several techniques and assembles to features
    However, it's focused on stream processing, transforming messages coming from LoRa gateway using the models
    from 'transform_models' dictionary, which are the models used for Scaler, PCA and SVD transformation, depending
    of the algorithm that is being used 
    
    """
    def features_assembler_stream(df, model_type, transform_models):
        # Asseble all attributes except DevAddr, intrusion and prediction that will not be used for model training, only to identify the model
        column_names = list(set(get_all_attributes_names(df.schema)) - set(["DevAddr", "intrusion"]))
        
        assembler = VectorAssembler(inputCols=column_names, outputCol="feat")
        df = assembler.transform(df)

        scaler_model = transform_models["StdScaler"]
        df = scaler_model.transform(df)

        if model_type in [ModelType.IF_CUSTOM, ModelType.LOF]:
            pca_model = transform_models["PCA"]
            df = pca_model.transform(df)

        elif model_type in [ModelType.OCSVM, ModelType.HBOS, ModelType.KNN]:
            V = transform_models["SVD"]

            # Manually applies the transformation feat * V to obtain the new reduced vectors
            def project_features(feat):
                # feat is a DenseVector from pyspark.ml.linalg, so feat.dot(V) works
                projected_array = feat.dot(V)
                return MLDenseVector(projected_array)
            
            project_udf = F.udf(project_features, returnType=VectorUDT())
            df = df.withColumn("features", project_udf("scaled"))

        else:
            df = df.withColumnRenamed("scaled", "features")

        return df.drop("feat", "scaled")
       
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
    Method to reverse hexadecimal octets in string format
    
        hex_str: hexadecimal value
    
    """
    @staticmethod
    def reverse_hex_bytes(hex_str):

        # If hex_str is None or an empty string, return -1
        if (hex_str is None) or (hex_str == ""):
            return hex_str
        
        # Divides hex_str into octets
        octets = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
        reversed_octets = "".join(reversed(octets))
        
        return reversed_octets

    """
    Method used to extract LoRaWAN parameters from data during stream processing
    
    """
    @staticmethod
    def parse_data(data):
        
        result = {
                    "AppEUI": None,
                    "AppNonce": None,
                    "DLSettings": None,
                    "CFList": None,
                    "CFListType": None,
                    "DevAddr": None,
                    "DevEUI": None,
                    "DevNonce": None,
                    "FCnt": None,
                    "FCtrl": None,
                    "FOpts": None,
                    "FPort": None,
                    "PHYPayloadLen": None,
                    "MIC": None,
                    "MHDR": None,
                    "RxDelay": None
                }

        # If data is inexistent, all resulting fields are also inexistent (None)
        if not data:
            return result
        
        # Decode to base64, convert to hexadecimal, remove spaces if any, and put letters in uppercase
        phypayload_hex = base64.b64decode(data).hex().replace(" ", "").upper()

        # Compute the size of PHYPayload; we only use its size because when PHYPayload is too large, Spark might not be able
        # to process it, since it's only able to process values with at most 38 digits
        result['PHYPayloadLen'] = len(phypayload_hex)

        # 12 characters (6 bytes) is the absolute minimum size of PHYPayload; PHYPayload is composed by:
            # MHDR: 1 byte
            # MIC: 4 bytes
            # MACPayload: 1...M bytes
        if len(phypayload_hex) < 12:
            return result

        # MIC (last 4 bytes)
        result['MIC'] = phypayload_hex[-8:]
        mic_offset = len(phypayload_hex) - 8

        # MHDR (first byte)
        mhdr = phypayload_hex[:2]
        result['MHDR'] = mhdr

        # Convert MHDR to binary and extract the first 3 bits that correspond to the message type (MType)
        # Fill the result with zeros to left to get exactly 8 bits on this case; and not 5 or 6;
        # In general, zfill guarantees that the conversion from hexadecimal to binary always results on a binary 
        # number size of 4 times the size of the hexadecimal
        mhdr_binary_original = format(int(mhdr, 16), 'b')
        mhdr_binary = mhdr_binary_original.zfill(len(mhdr) * 4)
        m_type = mhdr_binary[:3] 

        # MType = '000' -> Join Request
        if m_type == '000':   
            result['AppEUI'] = DataPreProcessing.reverse_hex_bytes(phypayload_hex[2:18])
            result['DevEUI'] = DataPreProcessing.reverse_hex_bytes(phypayload_hex[18:34])
            result['DevNonce'] = DataPreProcessing.reverse_hex_bytes(phypayload_hex[34:38])

        # MType = '001' -> Join Accept
        elif m_type == '001':            
            result['AppNonce'] = DataPreProcessing.reverse_hex_bytes(phypayload_hex[2:8])
            result['DevAddr'] = DataPreProcessing.reverse_hex_bytes(phypayload_hex[14:22])
            result['DLSettings'] = phypayload_hex[22:24]
            result['RxDelay'] = phypayload_hex[24:26]
            cf_list = phypayload_hex[26:mic_offset]

            # separate CFList on two, CFList maximum 15 bytes and CFListType maximum 1 byte, 
            # because of the limit of the size of values supported by Spark 
            # (38 digits of integer from 30 digit (15 bytes) hexadecimal)
            result['CFListType'] = cf_list[-2:]     # CFListType is the last byte of CFList
            result['CFList'] = cf_list[:-2]         # part of CFList that excludes CFListType

        # if Message Type (MType) is different from Join Request and Join Accept
        else:                   

            # FHDR: DevAddr (4 bytes), FCtrl (1 byte), FCnt (2 bytes); but all these fields also depends of the size of MACPayload,
            # which can be between 1 and M bytes
            result['DevAddr'] = DataPreProcessing.reverse_hex_bytes((phypayload_hex[:mic_offset])[2:10])
            fctrl = (phypayload_hex[:mic_offset])[10:12]
            result['FCtrl'] = fctrl
            result['FCnt'] = DataPreProcessing.reverse_hex_bytes((phypayload_hex[:mic_offset])[12:16])

            # FOpts and FPort depend of the second byte of FCtrl
            if len(fctrl) == 2:
                fctrl_byte_original = format(int(fctrl, 16), 'b')
                fctrl_byte = fctrl_byte_original.zfill(len(fctrl) * 4)

                # Apply AND operator to determine value of FOptsLen
                # For example: '110100' & '0100' = '0100'; this takes into consideration the value of the last 4 bits of FOpts as FOptsLen
                # FOptsLen is the size of FOpts in bytes (not in characters)
                fopts_len = int(fctrl_byte, 2) & 0x0F               

                fopts = (phypayload_hex[:mic_offset])[16:16 + (fopts_len * 2)]   # FOpts

                # Next field: FPort (0 or 1 byte, if exists)
                fport_index = 16 + len(fopts)
                result['FPort'] = (phypayload_hex[:mic_offset])[fport_index:fport_index + 2]
                result['FOpts'] = fopts

            # If FCtrl has no bytes, it's because MACPayload size is very small and, in that case, FOpts and FPort are also inexistent
            # If FCtrl has more than one byte, that's an anomaly that must be detected by ML models 
            else:
                result['FOpts'] = ""
                result['FPort'] = ""

        return result
    
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
    The limit of the length of hexadecimal attributes is 31 characters, which is 15 bytes; because thats the limit to result in
    a decimal number that does not exceed the length of 38 digits that Spark supports
    
        df: spark dataframe that represents the dataset
        attributes: a list of strings with the names of attributes to be converted
    
    """
    @staticmethod
    def hex_to_decimal(df, attributes):

        # NOT NULL: Converted number to decimal
        # NULL or EMPTY STRING: NULL
        # ANOMALY: -1
        def hex_to_decimal_int(hex_str):
    
            # If the attribute is absent (NULL or empty), just return None
            if hex_str is None or hex_str == "":
                return None
            
            try:
                result = int(hex_str, 16)      # Hexadecimal converted to decimal
                
                # When the size is higher than expected, indicate an anomaly through -1, indicating
                # that the attribute is too large, and also because Spark will not be able to process
                # values with more than 38 digits
                if len(str(result)) > 38:
                    return Decimal(-1)

                return Decimal(result)
            
            except:
                return Decimal(-1)
                
        hex_to_decimal_udf = F.udf(hex_to_decimal_int, DecimalType(38, 0))

        for attr in attributes: 

            # Fill missing values (None or empty strings) with None, and anomalies with -1, since -1
            # would never be a valid value for an hexadecimal-to-decimal attribute
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

