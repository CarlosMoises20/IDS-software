

from pyspark.sql.functions import expr, when, col, asc, concat, length
from preProcessing.pre_processing import DataPreProcessing
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import Imputer
from auxiliaryFunctions.general import get_all_attributes_names


class TxpkPreProcessing(DataPreProcessing):


    """

    This function applies pre-processing on data from the dataframe 'df_txpk', for the 'txpk' dataset

        - Applies feature selection techniques to remove irrelevant attributes (dimensionality reduction),
            selecting only the attributes that are relevant to build the intended model_numeric for IDS
            

        - Converts boolean attributes to integer attributes

        - Creates new useful attributes, such as "CFListType", that results from the last octet of "CFList" attribute 

        - Verifies if fields are correctly calculated, creating new attributes that indicate if the field is correctly calculated

        - Converts hexadecimal attributes to decimal


    """
    @staticmethod
    def pre_process_data(df):

        # Specify only the attributes to keep, and explode 'txpk' struct attribute to simplify processing
        # of attributes inside the 'txpk' struct
        selected_columns = [
            "AppNonce", "CFList", "DLSettings", "DLSettingsRX1DRoffset", 
            "DLSettingsRX2DataRate", "DevAddr", "FCnt", "FCtrl", 
            "FCtrlACK", "FHDR", "FOpts", "FPort", "FRMPayload",
            "FreqCh4", "FreqCh5", "FreqCh6", "FreqCh7", "FreqCh8", 
            "MACPayload", "MHDR", "MIC", "MessageType", "NetID", 
            "PHYPayload", "RxDelay", "txpk.*"
        ]

        # Select only the specified columns
        df = df.select(*selected_columns)

        # Remove irrelevant / redundant attributes that used to be inside 'txpk' array,
        # as well as attributes that have always the same value
        df = df.drop("codr", "imme", "ipol", "modu", "ncrc", "rfch")

        # create a new attribute called "CFListType", coming from the last octet of "CFList" according to the LoRaWAN v1.1 specification
        # source: https://lora-alliance.org/resource_hub/lorawan-specification-v1-1/ 
        df = df.withColumn("CFListType", expr("substring(CFList, -2, 2)"))

        # remove the "CFList" attribute, since it's already split to "FreqCh4", "FreqCh5", "FreqCh6", 
        # "FreqCh7", "FreqCh8" and "CFListType", for a more simple processing
        df = df.drop("CFList")
        
        # Convert MessageType parameter to its corresponding value in decimal
        df = df.withColumn("MessageType", when(col("MessageType") == "Join Request", 0)
                                          .when(col("MessageType") == "Join Accept", 1)
                                          .when(col("MessageType") == "Unconfirmed Data Up", 2)
                                          .when(col("MessageType") == "Unconfirmed Data Down", 3)
                                          .when(col("MessageType") == "Confirmed Data Up", 4)
                                          .when(col("MessageType") == "Confirmed Data Down", 5)
                                          .when(col("MessageType") == "Rejoin Request", 6)
                                          .when(col("MessageType") == "Proprietary", 7)
                                          .otherwise(-1))

        # Convert 'FCtrlACK' to integer: True = 1, False = 0; ensuring that an acknowledge was received is useful to detect anomalies
        # source: https://lora-alliance.org/resource_hub/lorawan-specification-v1-1/
        df = df.withColumn("FCtrlACK", when(col("FCtrlACK") == True, 1)
                                      .when(col("FCtrlACK") == False, 0)
                                      .otherwise(-1)) 

        # Create a udf to compare fields that correspond to part of 
        # others but with reversed octets
        reverse_hex_udf = udf(DataPreProcessing.reverse_hex_octets, StringType())

        # TODO: analyse if these 3 steps are really necessary
        df = df.withColumn("Valid_FHDR", when(col("FHDR").isNull(), -1)
                                            .when(col("FHDR") == concat(reverse_hex_udf(col("DevAddr")), 
                                                                        reverse_hex_udf(col("FCtrl")), 
                                                                        reverse_hex_udf(col("FCnt")), 
                                                                        col("FOpts")), 1)
                                            .otherwise(0))

        df = df.withColumn("Valid_MACPayload", when(col("MACPayload").isNull(), -1)
                                                .when(col("MACPayload") == concat(col("FHDR"), 
                                                                            col("FPort"), 
                                                                            col("FRMPayload")), 1)
                                                .otherwise(0))
        
        """
        df = df.withColumn("Valid_MHDR", when(col("MHDR").isNull(), -1)
                                          .when(hex_to_binary_udf(col("MHDR"))[:3] == bin(col("MessageType")), 1)
                                          .otherwise(0))
        """

        # After calculating RFU, remove DLSettings
        df = df.drop("DLSettings")

        # Create 'dataLen' that corresponds to the length of 'data', 
        # that represents the content of the LoRaWAN message
        df = df.withColumn("dataLen", length(col("data")))

        # TODO: check is "data" and "datr" are not needed
        df = df.drop("data", "datr")

        # manually define hexadecimal attributes from the 'df' dataframe, that are part
        # of the LoRaWAN specification
        hex_attributes = ["AppNonce", "CFListType", "DevAddr", "FCnt", 
                        "FCtrl", "FCtrlACK", "FHDR", "FOpts", 
                        "FPort", "FRMPayload", "FreqCh4", "FreqCh5", "FreqCh6",
                        "FreqCh7", "FreqCh8", "MACPayload", "MHDR", "MIC", "NetID",
                        "PHYPayload", "RxDelay"]

        # Convert hexadecimal attributes (string) to decimal (int), since these are values that are calculated
        # this also replaces NULL values with -1 to be supported by the algorithms
        # if we want to apply machine learning algorithms, we need numerical values and if these values stayed as strings,
        # these would be treated as categorical values, which is not the case
        df = DataPreProcessing.hex_to_decimal(df, hex_attributes)

        # get all other attributes that used not to be hexadecimal
        non_hex_attributes = list(set(get_all_attributes_names(df.schema)) - set(hex_attributes))
        
        # TODO: find an reasonable approach to impute missing values on the remaining attributes (e.g. the ones that were always numeric
        # and the categorical attributes)
        imputer = Imputer(inputCols=non_hex_attributes, outputCols=non_hex_attributes, strategy="mean")

        df = imputer.fit(df).transform(df)

        # TODO: during processing, analyse if it's necessary to apply some more pre-processing steps

        df = df.withColumn("intrusion", when((col("Valid_FHDR") == 1) & (col("Valid_MACPayload") == 1), 0).otherwise(1))


        return df

