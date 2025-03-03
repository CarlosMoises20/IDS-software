

from pyspark.sql.functions import struct, expr, when, col
from preProcessing.pre_processing import DataPreProcessing
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


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

        ## Feature Selection: remove irrelevant, redundant and correlated attributes
        df = df.drop("type", "recv_date", "fromip", "ip", "port",
                    "Direction", "RxDelayDel", "FCtrlADR")

        # apply filter to let pass only relevant attributes inside 'txpk' property of struct type
        df = df.withColumn("txpk", struct(
                                            expr("txpk.data AS data"),
                                            expr("txpk.datr AS datr"),
                                            expr("txpk.freq AS freq"),
                                            expr("txpk.powe AS powe"),
                                            expr("txpk.size AS size"),
                                            expr("txpk.tmst AS tmst")
                                        )
                           )

        # create a new attribute called "CFListType", coming from the last octet of "CFList" according to the LoRaWAN v1.1 specification
        # source: https://lora-alliance.org/resource_hub/lorawan-specification-v1-1/ 
        df = df.withColumn("CFListType", expr("substring(CFList, -2, 2)"))

        # remove the "CFList" attribute, since it's already split to "FreqCh4", "FreqCh5", "FreqCh6", 
        # "FreqCh7", "FreqCh8" and "CFListType", for a more simple processing
        df = df.drop("CFList")

        # Convert 'FCtrlACK' to integer: True = 1, False = 0; ensuring that an acknowledge was received is useful to detect anomalies
        # source: https://lora-alliance.org/resource_hub/lorawan-specification-v1-1/
        df = DataPreProcessing.bool_to_int(df, ["FCtrlACK"])

        # Define a UDF que usa a função reverse_hex_octets
        reverse_hex_udf = udf(DataPreProcessing.reverse_hex_octets, StringType())

        # TODO 1: continue to add columns that validate fields to check if they are correctly calculated
        """
        df = df.withColumn(
            "Valid_FHDR",
            when(col("FHDR").isNull(), None) 
            .when(col("FHDR") == reverse_hex_udf(col("DevAddr")) + 
                                reverse_hex_udf(col("FCtrl")) + 
                                reverse_hex_udf(col("FCnt")) + 
                                col("FOpts"), 1)
            .otherwise(0)
        )
        """

        df.select("FHDR", "DevAddr", "FCtrl", "FCnt", "FOpts", "valid_fhdr").dropDuplicates().show(30, truncate=False)


        # TODO 1A: calculate RFU, that comes from various fields


        # After calculating RFU, remove DLSettings
        df = df.drop("DLSettings")


        # TODO 1B: calculate MIC, that comes from: MHDR | FHDR | FPort | FRMPayload
        # TODO 1C: calculate other fields that are split by others



        # Convert hexadecimal attributes (string) to decimal (int)
        df = DataPreProcessing.hex_to_decimal(df, ["PHYPayload", "MHDR", "MACPayload",
                                                   "MIC", "FHDR", "FPort", "FRMPayload",
                                                   "DevAddr", "FCnt", "FCtrl", "FOpts",
                                                   "AppNonce", "NetID", "FreqCh4", "FreqCh5",
                                                   "FreqCh6", "FreqCh7", "FreqCh8", "CFListType"])


        # TODO: during processing, analyse if it's necessary to apply some more pre-processing steps

        return df

