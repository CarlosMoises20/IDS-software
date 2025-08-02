
from prepareData.preProcessing.pre_processing import DataPreProcessing

class TxpkPreProcessing(DataPreProcessing):

    """ NOTE: for stream processing this was not tested, because the LoRa gateway only sends RXPK and STATS messages
    This function applies pre-processing on data from the dataframe 'df_txpk', for the 'txpk' dataset

        - Applies feature selection techniques to remove irrelevant attributes (dimensionality reduction),
            selecting only the attributes that are relevant to build the intended model_numeric for IDS   

        - Converts boolean attributes to integer attributes

        - Creates new useful attributes, such as "CFListType", that results from the last octet of "CFList" attribute 

        - Verifies if fields are correctly calculated, creating new attributes that indicate if the field is correctly calculated

        - Converts hexadecimal attributes to decimal

    stream_processing is a boolean which indicates if the pre-processing is being done in the context of stream processing (i.e. 
    receiving messages in real-time from a LoRa gateway) or in the context of creating static models

    """
    @staticmethod
    def pre_process_data(df, stream_processing=False):

        # Stream processing for TXPK is not applicable on this project
        if stream_processing:
            pass

        else:

            # Specify only the attributes to keep, and explode 'txpk' struct attribute to simplify processing
            # of attributes inside the 'txpk' struct attribute
            selected_columns = [
                "AppNonce", "CFList", "DLSettings", "DevAddr", "FCtrl",
                "FCnt", "FOpts", "FPort", "PHYPayload", "MIC", "MHDR",
                "RxDelay", "txpk.*"
            ]

            # Select only the specified columns, removing irrelevant, redundant or correlated attributes
            df = df.select(*selected_columns)

            # Remove irrelevant attributes that used to be inside 'txpk' struct attribute before exploding 'txpk'
            df = df.drop("ipol", "modu", "imme", "ncrc", "data")

        return df

