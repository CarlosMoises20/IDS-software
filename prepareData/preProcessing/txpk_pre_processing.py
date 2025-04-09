

from pyspark.sql.functions import when, col
from prepareData.preProcessing.pre_processing import DataPreProcessing
from common.constants import *

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
        # of attributes inside the 'txpk' struct attribute
        selected_columns = [
            "AppNonce", "CFList", "DLSettingsRX1DRoffset", 
            "DLSettingsRX2DataRate", "DevAddr", "FCnt", "FCtrl", 
            "FCtrlACK", "FCtrlADR", "FOpts", "FPort", "FRMPayload",
            "FreqCh4", "FreqCh5", "FreqCh6", "FreqCh7", "FreqCh8", 
            "MIC", "MessageType", "NetID", "RxDelay", "txpk.*"
        ]

        # Select only the specified columns, removing irrelevant, redundant or correlated attributes
        df = df.select(*selected_columns)

        # Remove irrelevant attributes that used to be inside 'txpk' struct attribute
        df = df.drop("ipol", "modu")

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

        return df

