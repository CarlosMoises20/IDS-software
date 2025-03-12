
from preProcessing.pre_processing import DataPreProcessing
from pyspark.sql.functions import expr, col, explode, length, when, col, udf, concat, asc, desc
from pyspark.sql.types import StringType
from pyspark.ml.feature import Imputer
from auxiliaryFunctions.general import get_all_attributes_names

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
                                                            ELSE -1
                                                       END, 
                                        'NetID', x.NetID, 
                                        'PHYPayload', x.PHYPayload, 
                                        'RxDelay', x.RxDelay, 
                                        'chan', x.chan,
                                        'data', x.data, 
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

        # explode 'rxpk' array, since each element inside the 'rxpk' array corresponds to a different LoRaWAN message
        df = df.withColumn("rxpk", explode(col("rxpk")))

        # change attribute names to be recognized by PySpark ML algorithms (for example, 'rxpk.AppEUI' -> 'AppEUI')
        # this also removes all attributes outside the 'rxpk' array, since these are all irrelevant / redundant
        df = df.select("rxpk.*")

        # aggregate 'chan' and 'lsnr' arrays, removing NULL values
        df = df.withColumn("chan", when(col("rsig.chan").isNotNull() | col("chan").isNotNull(),
                                                expr("filter(array_union(coalesce(array(chan), array()), coalesce(rsig.chan, array())), x -> x IS NOT NULL AND x = x)")
                                        ).otherwise(None)
                ).withColumn("lsnr", when(col("rsig.lsnr").isNotNull() | col("lsnr").isNotNull(),
                                                expr("filter(array_union(coalesce(array(lsnr), array()), coalesce(rsig.lsnr, array())), x -> x IS NOT NULL AND x = x)")
                                        ).otherwise(None)
                )
        
        # split "chan" by "chan1" and "chan2" and "lsnr" by "lsnr1" and "lsnr2", since Vectors on algorithms
        # do not support arrays, only numeric values
        df = df.withColumn("chan1", col("chan")[0])   # the first element of 'chan' array
        df = df.withColumn("chan2", col("chan")[1])   # the second element of 'chan' array
        df = df.withColumn("lsnr1", col("lsnr")[0])   # the first element of 'lsnr' array
        df = df.withColumn("lsnr2", col("lsnr")[1])   # the second element of 'lsnr' array
        
        # remove 'rsig' array and 'chan' and 'lsnr' after aggregation and splitting of 'chan' and 'lsnr'
        df = df.drop("rsig", "chan", "lsnr")
        
        # Create a udf to compare fields that correspond to part of 
        # others but with reversed octets
        reverse_hex_udf = udf(DataPreProcessing.reverse_hex_octets, StringType())

        # TODO: analyse if these 3 steps are necessary
        
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
        """
        df = df.withColumn("Valid_MHDR", when(col("MHDR").isNull(), -1)
                                          .when(hex_to_binary_udf(col("MHDR"))[:3] == bin(col("MessageType")), 1)
                                          .otherwise(0))
        """

        
        # Create 'DataLen' that corresponds to the length of 'data', 
        # that represents the content of the LoRaWAN message
        df = df.withColumn("dataLen", length(col("data")))

        # remove 'data' after creating 'dataLen'
        df = df.drop("data")

        # manually define hexadecimal attributes from the 'df' dataframe, that are part
        # of the LoRaWAN specification
        hex_attributes = ["AppEUI", "AppNonce", "DevAddr", "DevEUI",
                        "DevNonce", "FCnt", "FCtrl", "FHDR",
                        "FOpts", "FPort", "FRMPayload", "MACPayload",
                        "MHDR", "MIC", "NetID", "PHYPayload", "RxDelay"]

        # Convert hexadecimal attributes (string) to numeric (DecimalType), replacing NULL values with -1 since
        # -1 would never be a valid value for an hexadecimal-to-decimal attribute
        df = DataPreProcessing.hex_to_decimal(df, hex_attributes)

        # get all non-hexadecimal attributes of the dataframe
        non_hex_attributes = list(set(get_all_attributes_names(df.schema)) - set(hex_attributes))
        
        # TODO: for the remaining attributes (that are all numeric), check if mean is really the best way 
        # to impute missing values 
        imputer = Imputer(inputCols=non_hex_attributes, outputCols=non_hex_attributes, strategy="mean")

        df = imputer.fit(df).transform(df)

        # TODO: after starting processing, analyse if it's necessary to apply some more pre-processing steps

        df = df.withColumn("intrusion", when((col("Valid_FHDR") == 1) & (col("Valid_MACPayload") == 1), 0).otherwise(1))

        return df

