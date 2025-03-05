
from preProcessing.pre_processing import DataPreProcessing
from pyspark.sql.functions import expr, col, explode, length, when, col, udf, concat
from pyspark.sql.types import StringType


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

        ## Feature Selection: remove irrelevant, redundant and correlated attributes
        # apply filter to let pass only relevant attributes inside 'rxpk' and 'rsig' arrays
        # that filter also removes more irrelevant, redundant and correlated attributes, as well as 
        # attributes that are always NULL
        df = df.withColumn("rxpk", expr("""
                                        transform(rxpk, x -> named_struct( 'AppEUI', x.AppEUI, 
                                        'AppNonce', x.AppNonce, 
                                        'DLSettings', x.DLSettings, 
                                        'DLSettingsRX1DRoffset', x.DLSettingsRX1DRoffset, 
                                        'DLSettingsRX2DataRate', x.DLSettingsRX2DataRate, 
                                        'DevAddr', x.DevAddr, 
                                        'DevEUI', x.DevEUI, 
                                        'DevNonce', x.DevNonce, 
                                        'FCnt', x.FCnt,
                                        'FCtrl', x.FCtrl,
                                        'FHDR', x.FHDR, 
                                        'FOpts', x.FOpts, 
                                        'FPort', x.FPort, 
                                        'FRMPayload', x.FRMPayload, 
                                        'MACPayload', x.MACPayload, 
                                        'MHDR', x.MHDR, 
                                        'MIC', x.MIC, 
                                        'MessageType', CASE 
                                                            WHEN x.MessageType = 'Join Request' THEN 0 
                                                            WHEN x.MessageType = 'Join Accept' THEN 1 
                                                            WHEN x.MessageType = 'Unconfirmed Data Up' THEN 2 
                                                            WHEN x.MessageType = 'Unconfirmed Data Down' THEN 3 
                                                            WHEN x.MessageType = 'Confirmed Data Up' THEN 4 
                                                            WHEN x.MessageType = 'Confirmed Data Down' THEN 5 
                                                            WHEN x.MessageType = 'Rejoin Request' THEN 6 
                                                            WHEN x.MessageType = 'Proprietary' THEN 7 
                                                            ELSE NULL
                                                       END, 
                                        'NetID', x.NetID, 
                                        'PHYPayload', x.PHYPayload, 
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
                                        'tmst', x.tmst ))
                                    """))
        
        # TODO: maybe remove 'rsig' later

        # explode 'rxpk' array, since each element inside the 'rxpk' array corresponds to a different LoRaWAN message
        df = df.withColumn("rxpk", explode(col("rxpk")))

        # change attribute names to be recognized by PySpark ML algorithms (for example, 'rxpk.AppEUI' -> 'AppEUI')
        # this also removes all attributes outside the 'rxpk' array, since these are all irrelevant / redundant
        df = df.select("rxpk.*")    
        
        # Create a udf to compare fields that correspond to part of 
        # others but with reversed octets
        reverse_hex_udf = udf(DataPreProcessing.reverse_hex_octets, StringType())

        # TODO 1: continue to add columns that validate fields to check if they are correctly calculated
        df = df.withColumn("Valid_FHDR", when(col("FHDR").isNull(), None)
                                          .when(col("FHDR") == concat(reverse_hex_udf(col("DevAddr")), 
                                                                        reverse_hex_udf(col("FCtrl")), 
                                                                        reverse_hex_udf(col("FCnt")), 
                                                                        col("FOpts")), 1)
                                           .otherwise(0))

        df = df.withColumn("Valid_MACPayload", when(col("MACPayload").isNull(), None)
                                                .when(col("MACPayload") == concat(col("FHDR"), 
                                                                            col("FPort"), 
                                                                            col("FRMPayload")), 1)
                                                .otherwise(0))
        

        #df.select("FHDR", "DevAddr", "FCtrl", "FCnt", "FOpts", "Valid_FHDR").dropDuplicates().show(30, truncate=False)

        # TODO: calculate RFU


        # After calculating RFU, remove DLSettings
        df = df.drop("DLSettings")
        
        # Create 'DataLen' that corresponds to the length of 'data', 
        # that represents the content of the LoRaWAN message
        df = df.withColumn("DataLen", length(col("data")))

        # TODO: verify later if these fields are not necessary
        df = df.drop("data", "datr")

        # Convert hexadecimal attributes (string) to decimal (int)
        df = DataPreProcessing.hex_to_decimal(df, ["AppEUI", "AppNonce", "DevAddr", "DevEUI",
                                                   "DevNonce", "FCnt", "FCtrl", "FHDR",
                                                   "FOpts", "FPort", "FRMPayload", "MACPayload",
                                                   "MHDR", "MIC", "NetID", "PHYPayload", "RxDelay"])
        
        df.select("codr").dropDuplicates().show(30, truncate=False)

        df = DataPreProcessing.str_to_float(df, ["codr"])

        df.select("codr").dropDuplicates().show(30, truncate=False)
        
        # Fill missing values with 0 for numeric attributes
        df = df.na.fill(0)
        
        # Fill missing values with "Unknown" for string attributes
        #df = df.na.fill("Unknown")
        
        # TODO: after starting processing, analyse if it's necessary to apply some more pre-processing steps


        return df

