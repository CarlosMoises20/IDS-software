

from pyspark.sql.functions import struct, expr, col, explode
from preProcessing.pre_processing import PreProcessing


class TxpkPreProcessing(PreProcessing):


    def __init__(self, df):
        super().__init__(df)



    """

    This function applies pre-processing on data from the dataframe 'df_txpk', for the 'txpk' dataset

    1 - Applies feature selection techniques to remove irrelevant attributes (dimensionality reduction),
            selecting only the attributes that are relevant to build the intended model_numeric for IDS 


    """
    def pre_process_dataset(self):

        ## Feature Selection: remove irrelevant, redundant and correlated attributes
        self.__df = self.__df.drop("type", "recv_date", "fromip", "ip", "port", 
                                    "Direction", "RxDelayDel", "DLSettings")

        # apply filter to let pass only relevant attributes inside 'txpk' 
        self.__df = self.__df.withColumn("txpk", struct(
                            expr("txpk.data AS data"),
                            expr("txpk.datr AS datr"),
                            expr("txpk.freq AS freq"),
                            expr("txpk.powe AS powe"),
                            expr("txpk.size AS size"),
                            expr("txpk.tmst AS tmst")
                            ))


        # create a new attribute called "CFListType", coming from the last octet of "CFList" according to the LoRaWAN v1.1 specification
        # source: https://lora-alliance.org/resource_hub/lorawan-specification-v1-1/ 
        self.__df = self.__df.withColumn("CFListType", expr("substring(CFList, -2, 2)"))

        # remove the "CFList" attribute, since it's already split to "FreqCh4", "FreqCh5", "FreqCh6", 
        # "FreqCh7", "FreqCh8" and "CFListType", for a more simple processing
        self.__df = self.__df.drop("CFList")


        # TODO: after starting processing, analyse if it's necessary to apply some more pre-processing steps

        return self.__df

