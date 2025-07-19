
from prepareData.preProcessing.pre_processing import DataPreProcessing
from pyspark.sql.functions import expr, col, explode, when, col, size, lit
from common.constants import *


"""
This class represents the pre-processing phase on LoRaWAN messages from the 'rxpk' dataset

"""
class RxpkPreProcessing(DataPreProcessing):

    """
    This method applies pre-processing on data from the dataframe 'df', for the 'rxpk' dataset

    1 - Applies feature selection techniques to remove irrelevant attributes (dimensionality reduction),
            selecting only the attributes that are relevant to build the intended model_numeric for IDS 

    """
    @staticmethod
    def pre_process_data(df, stream_processing=False):

        # TODO: add case of stream processing = True (decode "data" from base64, convert to hexadecimal and
        # to correspond to PHYPayload, and extract all relevant attributes from PHYPayload, depending of the message type

        ## Feature Selection: remove irrelevant, redundant and correlated attributes
        # apply filter to let pass only relevant attributes inside 'rxpk' and 'rsig' arrays
        # that filter also removes more irrelevant, redundant and correlated attributes, as well as 
        # attributes that are always NULL
        df = df.withColumn("rxpk", expr("""
                                        transform(rxpk, x -> named_struct('AppEUI', x.AppEUI, 
                                                                        'AppNonce', x.AppNonce, 
                                                                        'DLSettings', x.DLSettings,
                                                                        'CFList', x.CFList,  
                                                                        'DevAddr', x.DevAddr, 
                                                                        'DevEUI', x.DevEUI,
                                                                        'DevNonce', x.DevNonce, 
                                                                        'FCnt', x.FCnt,
                                                                        'FCtrl', x.FCtrl,
                                                                        'FOpts', x.FOpts, 
                                                                        'FPort', x.FPort, 
                                                                        'PHYPayload', x.PHYPayload,
                                                                        'MIC', x.MIC,
                                                                        'MHDR', x.MHDR,
                                                                        'RxDelay', x.RxDelay, 
                                                                        'chan', x.chan,
                                                                        'codr', x.codr,
                                                                        'datr', x.datr, 
                                                                        'freq', x.freq, 
                                                                        'lsnr', x.lsnr, 
                                                                        'rfch', x.rfch, 
                                                                        'rsig', transform(x.rsig, rs -> named_struct( 
                                                                                        'chan', rs.chan, 
                                                                                        'lsnr', rs.lsnr)),
                                                                        'rssi', x.rssi, 
                                                                        'size', x.size, 
                                                                        'stat', x.stat, 
                                                                        'tmst', x.tmst ))
                                                                    """))

        # explode 'rxpk' array, since each element inside the 'rxpk' array corresponds to a different LoRaWAN message
        df = df.withColumn("rxpk", explode(col("rxpk")))

        # change attribute names to be recognized by PySpark ML algorithms (for example, 'rxpk.AppEUI' -> 'AppEUI')
        # this also removes all attributes outside the 'rxpk' array, since these are all irrelevant / redundant
        df = df.select("rxpk.*")

        ### Extract "chan" and "lsnr" from "rsig" attribute

        # aggregate 'chan' and 'lsnr' arrays, removing NULL values
        df = df.withColumn("chan", when(col("rsig.chan").isNotNull() | col("chan").isNotNull(),
                                            expr("filter(array_union(coalesce(array(chan), array()), coalesce(rsig.chan, array())), x -> x IS NOT NULL AND x = x)")
                                        ).otherwise(None)) \
                .withColumn("lsnr", when(col("rsig.lsnr").isNotNull() | col("lsnr").isNotNull(),
                                            expr("filter(array_union(coalesce(array(lsnr), array()), coalesce(rsig.lsnr, array())), x -> x IS NOT NULL AND x = x)")
                                        ).otherwise(None))
        
        # split "chan" by "chan1" and "chan2" and "lsnr" by "lsnr1" and "lsnr2", since Vectors on algorithms
        # do not support arrays, only numeric values
        df = df.withColumn("chan1", when(size(col("chan")) > 0, col("chan")[0]).otherwise(lit(None)))
        df = df.withColumn("chan2", when(size(col("chan")) > 1, col("chan")[1]).otherwise(lit(None)))
        df = df.withColumn("lsnr1", when(size(col("lsnr")) > 0, col("lsnr")[0]).otherwise(lit(None)))
        df = df.withColumn("lsnr2", when(size(col("lsnr")) > 1, col("lsnr")[1]).otherwise(lit(None)))

        # remove 'rsig' array and 'chan' and 'lsnr' after aggregation and splitting of 'chan' and 'lsnr'
        df = df.drop("rsig", "chan", "lsnr")

        # convert attributes from hexadecimal to decimal that only exist in RXPK
        df = DataPreProcessing.hex_to_decimal(df, ["AppEUI", "DevEUI", "DevNonce"])

        return df

