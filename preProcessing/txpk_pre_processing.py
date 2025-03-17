
import time
from pyspark.sql.functions import expr, when, col, asc, concat, length, regexp_extract
from preProcessing.pre_processing import DataPreProcessing
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.feature import Imputer
from auxiliaryFunctions.general import get_all_attributes_names, format_time
from auxiliaryFunctions.anomaly_detection import AnomalyDetection
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
        # of attributes inside the 'txpk' struct
        selected_columns = [
            "AppNonce", "CFList", "DLSettingsRX1DRoffset", 
            "DLSettingsRX2DataRate", "DevAddr", "FCnt", "FCtrl", 
            "FCtrlACK", "FCtrlADR", "FHDR", "FOpts", "FPort", "FRMPayload",
            "FreqCh4", "FreqCh5", "FreqCh6", "FreqCh7", "FreqCh8", 
            "MACPayload", "MHDR", "MIC", "MessageType", "NetID", 
            "PHYPayload", "RxDelay", "txpk.*"
        ]

        # Select only the specified columns, removing irrelevant, redundant or correlated attributes
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


        ### Convert "FCtrlADR" and "FCtrlACK" attributes to integer

        df = df.withColumn("FCtrlADR", when(col("FCtrlADR") == True, 1)
                                        .when(col("FCtrlADR") == False, 0)
                                        .otherwise(-1)) \
                .withColumn("FCtrlACK", when(col("FCtrlACK") == True, 1)
                                        .when(col("FCtrlACK") == False, 0)
                                        .otherwise(-1))

        # Create a udf to compare fields that correspond to part of 
        # others but with reversed octets
        reverse_hex_udf = udf(DataPreProcessing.reverse_hex_octets, StringType())

        # TODO: analyse if these steps are really necessary
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

        # Create 'dataLen' that corresponds to the length of 'data', 
        # that represents the content of the LoRaWAN message
        df = df.withColumn("dataLen", length(col("data")))

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
        hex_attributes = ["AppNonce", "CFListType", "FCnt", 
                        "FCtrl", "FCtrlACK", "FHDR", "FOpts", 
                        "FPort", "FRMPayload", "FreqCh4", "FreqCh5", "FreqCh6",
                        "FreqCh7", "FreqCh8", "MACPayload", "MHDR", "MIC", "NetID",
                        "PHYPayload", "RxDelay"]

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


        # TODO: fix
        # define the label "intrusion" based on the result of the intrusion detection; this label will
        # be used for supervised learning of the models during training
        # Define "intrusion" based on MessageType without using UDFs
        df = df.withColumn("intrusion", when((col("MessageType") == 0) | (col("MessageType") == 6),  # Join Request and Rejoin-Request
                                            when((col("rssi") < RSSI_MIN) | (col("rssi") > RSSI_MAX), 1)  # Jamming
                                            .when((col("lsnr1") < LSNR_MIN) | (col("lsnr1") > LSNR_MAX), 1)
                                            .when((col("lsnr2") < LSNR_MIN) | (col("lsnr2") > LSNR_MAX), 1)
                                            .when(col("Valid_MACPayload") == 0, 1)  # Downlink Routing Attack
                                            .when(col("Valid_FHDR") == 0, 1)  # Physical Tampering
                                            .otherwise(0)
                                        ).when(
                                            col("MessageType") == 1,      # Join Accept
                                            when(col("Valid_MACPayload") == 0, 1)       # Downlink Routing Attack
                                            .otherwise(0)
                                        ).when(
                                            col("MessageType") == 2,  # Unconfirmed Data Up
                                            when(col("Valid_FHDR") == 0, 1)             # Physical Tampering
                                            .otherwise(0)
                                        ).when(
                                            col("MessageType") == 3,  # Unconfirmed Data Down
                                            when((col("rssi") < RSSI_MIN) | (col("rssi") > RSSI_MAX), 1)
                                            .when((col("Valid_MACPayload") == 0), 1)
                                            .otherwise(0)
                                        ).when(
                                            col("MessageType") == 4,  # Confirmed Data Up
                                            when((col("lsnr1") < LSNR_MIN) | (col("lsnr1") > LSNR_MAX), 1)
                                            .when((col("lsnr2") < LSNR_MIN) | (col("lsnr2") > LSNR_MAX), 1)
                                            .otherwise(0)
                                        ).when(
                                            col("MessageType") == 5,  # Confirmed Data Down
                                            when(col("Valid_FHDR") == 0, 1)
                                            .when(col("Valid_MACPayload") == 0, 1)
                                            .otherwise(0)
                                        ).when(
                                            col("MessageType") == 7,  # Proprietary
                                            when(col("Valid_FHDR") == 0, 1)
                                            .when(col("Valid_MACPayload") == 0, 1)
                                            .otherwise(0)
                                        ).otherwise(0)  # No MessageType, no intrusion
                                    )


        # apply normalization
        #df = DataPreProcessing.normalization(df)

        end_time = time.time()

        # Print the total time of pre-processing; the time is in seconds, minutes or hours
        print("Time of txpk pre-processing:", format_time(end_time - start_time), "\n\n")

        return df

