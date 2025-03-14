
import time
from pyspark.sql.functions import expr, when, col, asc, concat, length, struct
from preProcessing.pre_processing import DataPreProcessing
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.feature import Imputer
from auxiliaryFunctions.general import get_all_attributes_names, format_time
from auxiliaryFunctions.anomaly_detection import AnomalyDetection
from constants import *

class TxpkPreProcessing(DataPreProcessing):


    # this function will be fundamental to define the label that will be used as
    # the desired output in the model training, that will allow the model to fit
    # its weights and pendors using the input and the desired output
    @staticmethod
    def intrusion_detection(df_row):

        try:

            # TODO: introduce all types of possible LoRaWAN attacks here
            # TODO: correct the logic; separate examples for each device (DevAddr)

            pass

        except Exception as e:
            return 0  # Default to no intrusion if an error occurs



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
            "FCtrlACK", "FHDR", "FOpts", "FPort", "FRMPayload",
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

        # Convert 'FCtrlACK' to integer: True = 1, False = 0; ensuring that an acknowledge was received is useful to detect anomalies
        # source: https://lora-alliance.org/resource_hub/lorawan-specification-v1-1/
        df = df.withColumn("FCtrlACK", when(col("FCtrlACK") == True, 1)
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
        # this also replaces NULL and empty values with -1 to be supported by the algorithms
        # if we want to apply machine learning algorithms, we need numerical values and if these values stayed as strings,
        # these would be treated as categorical values, which is not the case
        df = DataPreProcessing.hex_to_decimal(df, hex_attributes)

        # get all other attributes that used not to be hexadecimal
        non_hex_attributes = list(set(get_all_attributes_names(df.schema)) - set(hex_attributes))
        
        # for the other numeric attributes, replace NULL and empty values with the mean, because these are values
        # that can assume any numeric value, so it's not a good approach to replace missing values with a static value
        # the mean is the best approach to preserve the distribution and variety of the data
        imputer = Imputer(inputCols=non_hex_attributes, outputCols=non_hex_attributes, strategy="mean")

        df = imputer.fit(df).transform(df)

        # Define UDF to apply the function of intrusion detection
        intrusion_udf = udf(lambda row: TxpkPreProcessing.intrusion_detection(row), IntegerType())

        # define the label "intrusion" based on the result of the intrusion detection; this label will
        # be used for supervised learning of the models during training
        df = df.withColumn("intrusion", intrusion_udf(struct(*df.columns)))

        # apply normalization
        #df = DataPreProcessing.normalization(df)

        end_time = time.time()

        # Print the total time of pre-processing; the time is in seconds, minutes or hours
        print("Time of txpk pre-processing: ", format_time(end_time - start_time), "\n\n")

        return df

