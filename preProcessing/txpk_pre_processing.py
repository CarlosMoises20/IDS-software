
import time
from pyspark.sql.functions import expr, when, col, concat, length, regexp_extract
from preProcessing.pre_processing import DataPreProcessing
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.feature import Imputer
from auxiliaryFunctions.general import get_all_attributes_names, format_time
from constants import *

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

        start_time = time.time()

        # Specify only the attributes to keep, and explode 'txpk' struct attribute to simplify processing
        # of attributes inside the 'txpk' struct attribute
        selected_columns = [
            "AppNonce", "CFList", "DLSettingsRX1DRoffset", 
            "DLSettingsRX2DataRate", "DevAddr", "DevEUI", "FCnt", "FCtrl", 
            "FCtrlACK", "FCtrlADR", "FOpts", "FPort", "FRMPayload",
            "FreqCh4", "FreqCh5", "FreqCh6", "FreqCh7", "FreqCh8", 
            "MHDR", "MIC", "MessageType", "NetID", 
            "RxDelay", "txpk.*"
        ]

        # Select only the specified columns, removing irrelevant, redundant or correlated attributes
        df = df.select(*selected_columns)

        # Remove irrelevant / redundant attributes that used to be inside 'txpk' struct attribute,
        # as well as attributes that have always the same value
        df = df.drop("codr", "imme", "ipol", "modu", "ncrc", "rfch")

        # create a new attribute called "CFListType", coming from the last octet of "CFList" according to the LoRaWAN specification
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
                                          .when(col("MessageType") == "RFU", 6)
                                          .when(col("MessageType") == "Proprietary", 7)
                                          .otherwise(-1))

        # Remove rows with invalid DevAddr and MessageType
        df = df.filter((col("DevAddr").isNotNull()) & (col("DevAddr") != "") & (col("MessageType") != -1))

        ### Convert "FCtrlADR" and "FCtrlACK" attributes to integer
        df = df.withColumn("FCtrlADR", when(col("FCtrlADR") == True, 1)
                                        .when(col("FCtrlADR") == False, 0)
                                        .otherwise(-1)) \
                .withColumn("FCtrlACK", when(col("FCtrlACK") == True, 1)
                                        .when(col("FCtrlACK") == False, 0)
                                        .otherwise(-1))

        # Create 'dataLen' that corresponds to the length of 'data', 
        # that represents the content of the LoRaWAN message
        df = df.withColumn("dataLen", length(col("data")))

        # remove 'data' after creating 'dataLen'
        # TODO: check if "data" is not needed
        df = df.drop("data")

        # regex pattern to extract "SF" and "BW" LoRa parameters from "datr"
        pattern = r"SF(\d+)BW(\d+)"

        # extract SF and BW from 'datr' attribute
        df = df.withColumn("SF", regexp_extract(col("datr"), pattern, 1).cast(IntegerType())) \
                .withColumn("BW", regexp_extract(col("datr"), pattern, 2).cast(IntegerType()))

        # Remove "datr" after splitting it by "SF" and "BW"
        df = df.drop("datr")

        # manually define hexadecimal attributes from the 'df' dataframe that will be
        # converted to decimal to be processed by the algorithms as values
        hex_attributes = ["AppNonce", "CFListType", "FCnt", "DevEUI",
                        "FCtrl", "FCtrlACK", "FOpts", 
                        "FPort", "FRMPayload", "FreqCh4", "FreqCh5", "FreqCh6",
                        "FreqCh7", "FreqCh8", "MHDR", "MIC", "NetID",
                        "RxDelay"]

        # Convert hexadecimal attributes (string) to decimal (int), since these are values that are calculated
        # this also replaces NULL and empty values with -1 to be supported by the algorithms
        # if we want to apply machine learning algorithms, we need numerical values and if these values stayed as strings,
        # these would be treated as categorical values, which is not the case
        df = DataPreProcessing.hex_to_decimal(df, hex_attributes)

        # get all other attributes
        remaining_attributes = list(set(get_all_attributes_names(df.schema)) - set(hex_attributes + ["DevAddr"]))
        
        # for the other numeric attributes, replace NULL and empty values with the mean, because these are values
        # that can assume any numeric value, so it's not a good approach to replace missing values with a static value
        # the mean is the best approach to preserve the distribution and variety of the data
        imputer = Imputer(inputCols=remaining_attributes, outputCols=remaining_attributes, strategy="mean")

        df = imputer.fit(df).transform(df)


        # apply normalization (TODO: check if this is necessary, maybe it will be)
        df = DataPreProcessing.normalization(df)

        end_time = time.time()

        # Print the total time of pre-processing; the time is in seconds, minutes or hours
        print("Time of txpk pre-processing:", format_time(end_time - start_time), "\n\n")

        return df

