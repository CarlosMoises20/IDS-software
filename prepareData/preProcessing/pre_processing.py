
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
        df_test = scaler_model.transform(df_test)

        fr_model = None
        
        if model_type in [ModelType.IF_CUSTOM, ModelType.LOF]:

            ### PCA (Principal Component Analysis)
                
            # Fit PCA using the train dataset    
            pca = PCA(k=len(column_names), inputCol="scaled", outputCol="features")
            pca_model = pca.fit(df_train)
            fr_model = pca_model
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

            project_udf = F.udf(project_features, returnType=VectorUDT())
            df_train = df_train.withColumn("features", project_udf("scaled"))
            df_test = df_test.withColumn("features", project_udf("scaled"))

        else:
            df_train = df_train.withColumnRenamed("scaled", "features")
            df_test = df_test.withColumnRenamed("scaled", "features")

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
    Method to reverse hexadecimal octets in string format
    
        hex_str: hexadecimal value
    
    """
    @staticmethod
    def reverse_hex_bytes(hex_str):

        # If hex_str is None or an empty string, return -1
        if (hex_str is None) or (hex_str == ""):
            return None
        
        # Divides hex_str into octets
        octets = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
        reversed_octets = "".join(reversed(octets))
        
        return reversed_octets

    """
    Method used to extract parameters from data during stream processing
    
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

        if not data:
            return result
        
        # Decode to base64, convert to hexadecimal, remove spaces if any, and put letters in uppercase
        phypayload_hex = base64.b64decode(data).hex().replace(" ", "").upper()

        print("data:", data)
        print("PHYPayload:", phypayload_hex)

        # 12 characters (6 bytes) is the absolute minimum size of PHYPayload
        # MHDR: 1 byte
        # MIC: 4 bytes
        # MACPayload: 1...M bytes
        if len(phypayload_hex) < 12:
            return result
        
        result['PHYPayloadLen'] = len(phypayload_hex)

        # MIC (last 4 bytes)
        result['MIC'] = phypayload_hex[-8:]
        mic_offset = len(phypayload_hex) - 8

        # MHDR (first byte)
        mhdr = phypayload_hex[:2]
        result['MHDR'] = mhdr

        # Convert MHDR to binary and extract the first 3 bits that correspond to the message type
        mhdr_binary = format(int(mhdr, 16), 'b')
        m_type = mhdr_binary[:3] 

        if m_type == '000':   # MType = '000' -> Join Request
            
            result['AppEUI'] = DataPreProcessing.reverse_hex_bytes(phypayload_hex[2:18])
            result['DevEUI'] = DataPreProcessing.reverse_hex_bytes(phypayload_hex[18:36])
            result['DevNonce'] = DataPreProcessing.reverse_hex_bytes(phypayload_hex[36:40])

        elif m_type == '001':   # MType = '001' -> Join Accept
            
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

        # if Message is different from Join Request and Join Accept
        else:                   

            # FHDR: DevAddr (4 bytes), FCtrl (1 byte), FCnt (2 bytes)
            result['DevAddr'] = DataPreProcessing.reverse_hex_bytes(phypayload_hex[2:10])
            fctrl = phypayload_hex[10:12]
            result['FCnt'] = DataPreProcessing.reverse_hex_bytes(phypayload_hex[12:16])

            # Size of FOpts depends of the second byte of FCtrl
            if fctrl:
                fctrl_byte = int(fctrl, 16)

                # Apply AND operator to determine value of FOptsLen
                # For example: '110100' & '0100' = '0100'; this takes into consideration the value of the last 4 bits of FOpts as FOptsLen
                # FOptsLen is the size of FOpts in bytes (not in characters)
                fopts_len = fctrl_byte & 0x0F                   

                fopts = phypayload_hex[16:16 + fopts_len * 2]   # FOpts

                # Next field: FPort (0 or 1 byte, if exists)
                fport_index = 16 + len(fopts)
                if fport_index >= mic_offset:
                    result['FPort'] = ""

                result['FPort'] = phypayload_hex[fport_index:fport_index + 2]
                result['FOpts'] = DataPreProcessing.reverse_hex_bytes(fopts)
                result['FCtrl'] = fctrl
            
            else:
                result['FCtrl'] = ""
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
    
            if hex_str is None or hex_str == "":
                return None
            
            try:
                res = int(hex_str, 16)      # Hexadecimal converted to decimal
                
                # When the size is higher than expected, indicate an anomaly through -1
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

