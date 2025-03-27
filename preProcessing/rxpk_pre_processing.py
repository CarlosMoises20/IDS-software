
import time
from preProcessing.pre_processing import DataPreProcessing
from pyspark.sql.functions import expr, col, explode, length, when, col, udf, regexp_extract, lit, asc
from pyspark.sql.types import StringType, IntegerType, FloatType
from pyspark.ml.feature import Imputer
from common.auxiliary_functions import get_all_attributes_names, format_time
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
    def pre_process_data(df):

        start_time = time.time()

        ## Feature Selection: remove irrelevant, redundant and correlated attributes
        # apply filter to let pass only relevant attributes inside 'rxpk' and 'rsig' arrays
        # that filter also removes more irrelevant, redundant and correlated attributes, as well as 
        # attributes that are always NULL
        df = df.withColumn("rxpk", expr("""
                                        transform(rxpk, x -> named_struct( 'AppEUI', x.AppEUI, 
                                        'AppNonce', x.AppNonce, 
                                        'DLSettingsRX1DRoffset', x.DLSettingsRX1DRoffset, 
                                        'DLSettingsRX2DataRate', x.DLSettingsRX2DataRate,
                                        'CFList', x.CFList,  
                                        'DevAddr', x.DevAddr, 
                                        'DevEUI', x.DevEUI,
                                        'DevNonce', x.DevNonce, 
                                        'FCnt', x.FCnt,
                                        'FCtrl', x.FCtrl,
                                        'FCtrlACK', x.FCtrlACK,
                                        'FCtrlADR', x.FCtrlADR,
                                        'FOpts', x.FOpts, 
                                        'FPort', x.FPort, 
                                        'FRMPayload', x.FRMPayload,
                                        'MIC', x.MIC, 
                                        'MessageType', CASE 
                                                            WHEN x.MessageType = 'Join Request' THEN 0 
                                                            WHEN x.MessageType = 'Join Accept' THEN 1 
                                                            WHEN x.MessageType = 'Unconfirmed Data Up' THEN 2 
                                                            WHEN x.MessageType = 'Unconfirmed Data Down' THEN 3 
                                                            WHEN x.MessageType = 'Confirmed Data Up' THEN 4 
                                                            WHEN x.MessageType = 'Confirmed Data Down' THEN 5 
                                                            WHEN x.MessageType = 'RFU' THEN 6 
                                                            WHEN x.MessageType = 'Proprietary' THEN 7 
                                                            ELSE -1
                                                       END, 
                                        'NetID', x.NetID, 
                                        'RxDelay', x.RxDelay, 
                                        'chan', x.chan,
                                        'codr', x.codr,
                                        'data', x.data,
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

        # Replace NULL and empty-string values of DevAddr with "Unknown"
        df = df.withColumn("DevAddr", when((col("DevAddr").isNull()) | (col("DevAddr") == lit("")), "Unknown")
                                        .otherwise(col("DevAddr")))
        
        ### Convert boolean attributes to integer values
        df = DataPreProcessing.bool_to_int(df, ["FCtrlADR", "FCtrlACK"])

        # Convert "codr" from string to float
        str_float_udf = udf(DataPreProcessing.str_to_float, FloatType())
        df = df.withColumn("codr", str_float_udf(col("codr")))

        ### Extract SF and BW from "datr" attribute
        pattern = r"SF(\d+)BW(\d+)"     # regex pattern to extract "SF" and "BW" 

        df = df.withColumn("SF", regexp_extract(col("datr"), pattern, 1).cast(IntegerType())) \
                .withColumn("BW", regexp_extract(col("datr"), pattern, 2).cast(IntegerType()))

        # Remove "datr" after splitting it by "SF" and "BW"
        df = df.drop("datr")


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
        df = df.withColumn("chan1", col("chan")[0])   # the first element of 'chan' array
        df = df.withColumn("chan2", col("chan")[1])   # the second element of 'chan' array
        df = df.withColumn("lsnr1", col("lsnr")[0])   # the first element of 'lsnr' array
        df = df.withColumn("lsnr2", col("lsnr")[1])   # the second element of 'lsnr' array
        
        # remove 'rsig' array and 'chan' and 'lsnr' after aggregation and splitting of 'chan' and 'lsnr'
        df = df.drop("rsig", "chan", "lsnr")

        # Create 'dataLen' and 'FRMPayload_Len' attributes that correspond to the length of 'data' and 'FRMPayload', 
        # that represents the content of the LoRaWAN message
        df = df.withColumn("dataLen", length(col("data"))) \
                .withColumn("FRMPayload_Len", length(col("FRMPayload")))

        # remove 'data' and 'FRMPayload' after computing their lengths
        # TODO: check if these are not needed
        df = df.drop("data", "FRMPayload")

        # create new attributes resulting from "CFList" division
        reverse_hex_octets_udf = udf(DataPreProcessing.reverse_hex_octets, StringType())

        df = df.withColumn("FreqCh4", reverse_hex_octets_udf(expr("substring(CFList, 1, 6)"))) \
                .withColumn("FreqCh5", reverse_hex_octets_udf(expr("substring(CFList, 7, 6)"))) \
                .withColumn("FreqCh6", reverse_hex_octets_udf(expr("substring(CFList, 13, 6)"))) \
                .withColumn("FreqCh7", reverse_hex_octets_udf(expr("substring(CFList, 19, 6)"))) \
                .withColumn("FreqCh8", reverse_hex_octets_udf(expr("substring(CFList, 25, 6)"))) \
                .withColumn("CFListType", when((col("CFList").isNull()) | (col("CFList") == ""), -1)
                                            .otherwise(expr("substring(CFList, 31, 2)")))

        # Remove "CFList" after splitting it
        df = df.drop("CFList")


        # manually define hexadecimal attributes from the 'df' dataframe that will be
        # converted to decimal to be processed by the algorithms as values
        hex_attributes = ["AppEUI", "AppNonce", "FreqCh4", "FreqCh5", 
                          "FreqCh6", "FreqCh7", "FreqCh8",
                          "CFListType", "DevEUI", "DevNonce",
                          "FCnt", "FCtrl", "FOpts", "FPort", 
                          "MIC", "NetID", "RxDelay"]
    
        # Convert hexadecimal attributes (string) to numeric (DecimalType), replacing NULL and empty values with -1 since
        # -1 would never be a valid value for an hexadecimal-to-decimal attribute
        df = DataPreProcessing.hex_to_decimal(df, hex_attributes)

        # get all other attributes of the dataframe
        remaining_attributes = list(set(get_all_attributes_names(df.schema)) - set(hex_attributes + ["DevAddr"]))
        
        # for the other numeric attributes, replace NULL and empty values with the mean, because these are values
        # that can assume any numeric value, so it's not a good approach to replace missing values with a static value
        # the mean is the best approach to preserve the distribution and variety of the data
        imputer = Imputer(inputCols=remaining_attributes, outputCols=remaining_attributes, strategy="mean")

        df = imputer.fit(df).transform(df)
    

        # apply normalization (TODO: check if this is necessary, maybe it will be)
        #df = DataPreProcessing.normalization(df)

        end_time = time.time()

        print("Time of rxpk pre-processing:", format_time(end_time - start_time), "\n\n")

        return df

