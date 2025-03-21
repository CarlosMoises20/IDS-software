
import time
from preProcessing.pre_processing import DataPreProcessing
from pyspark.sql.functions import expr, col, explode, length, when, col, udf, concat, regexp_extract
from pyspark.sql.types import StringType, IntegerType, BooleanType, DoubleType, NumericType
from pyspark.ml.feature import Imputer
from auxiliaryFunctions.general import get_all_attributes_names, format_time
from auxiliaryFunctions.anomaly_detection import AnomalyDetection
from constants import *


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
                                        'DevAddr', x.DevAddr, 
                                        'DevEUI', x.DevEUI, 
                                        'DevNonce', x.DevNonce, 
                                        'FCnt', x.FCnt,
                                        'FCtrl', x.FCtrl,
                                        'FCtrlACK', x.FCtrlACK,
                                        'FCtrlADR', x.FCtrlADR,
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

        # explode 'rxpk' array, since each element inside the 'rxpk' array corresponds to a different LoRaWAN message
        df = df.withColumn("rxpk", explode(col("rxpk")))

        # change attribute names to be recognized by PySpark ML algorithms (for example, 'rxpk.AppEUI' -> 'AppEUI')
        # this also removes all attributes outside the 'rxpk' array, since these are all irrelevant / redundant
        df = df.select("rxpk.*")

        # Remove rows with invalid DevAddr and MessageType
        df = df.filter(col("DevAddr").isNotNull() & (col("DevAddr") != "") & col("MessageType") != -1)


        ### Convert "FCtrlADR" and "FCtrlACK" attributes to integer values
        df = df.withColumn("FCtrlADR", when(col("FCtrlADR") == True, 1)
                                        .when(col("FCtrlADR") == False, 0)
                                        .otherwise(-1)) \
                .withColumn("FCtrlACK", when(col("FCtrlACK") == True, 1)
                                        .when(col("FCtrlACK") == False, 0)
                                        .otherwise(-1))


        ### Extract SF and BW from "datr" attribute

        # regex pattern to extract "SF" and "BW" LoRa parameters from "datr"
        pattern = r"SF(\d+)BW(\d+)"

        # extract SF and BW from 'datr' attribute
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


        ### Create 'Valid_FHDR', 'Valid_MACPayload' to check if "FHDR" and "MACPayload" are correctly calculated
        
        # Create a udf to compare fields that correspond to part of 
        # others but with reversed octets
        reverse_hex = udf(DataPreProcessing.reverse_hex_octets, StringType())

        # TODO: analyse if these steps are necessary
        df = df.withColumn("Valid_FHDR", when((col("FHDR").isNull()) | (col("DevAddr").isNull()) | 
                                              (col("FCtrl").isNull()) | (col("FCnt").isNull()) | 
                                              (col("FOpts").isNull()), -1)
                                          .when(col("FHDR") == concat(reverse_hex(col("DevAddr")), 
                                                                        reverse_hex(col("FCtrl")), 
                                                                        reverse_hex(col("FCnt")), 
                                                                        col("FOpts")), 1)
                                           .otherwise(0))

        df = df.withColumn("Valid_MACPayload", when((col("MACPayload").isNull()) | (col("FHDR").isNull()) |
                                                    (col("FPort").isNull()) | (col("FRMPayload").isNull()), -1)
                                                .when(col("MACPayload") == concat(col("FHDR"), 
                                                                            col("FPort"), 
                                                                            col("FRMPayload")), 1)
                                                .otherwise(0))

        
        # Create 'DataLen' that corresponds to the length of 'data', 
        # that represents the content of the LoRaWAN message
        df = df.withColumn("dataLen", length(col("data")))

        # remove 'data' after creating 'dataLen'
        # TODO: check if "data" is not needed
        df = df.drop("data")

        # manually define hexadecimal attributes from the 'df' dataframe that will be
        # converted to decimal to be processed by the algorithms as values
        hex_attributes = ["AppEUI", "AppNonce", "DevEUI",
                        "DevNonce", "FCnt", "FCtrl", "FHDR",
                        "FOpts", "FPort", "FRMPayload", "MACPayload",
                        "MHDR", "MIC", "NetID", "PHYPayload", "RxDelay"]
        
        # Show attributes to see how some "labels" behave
        df.select("DevAddr", "DevEUI", "FHDR", "FPort", "FRMPayload", "MACPayload", "Valid_MACPayload") \
            .show(50, truncate=False)

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

        # TODO: fix
        # define the label "intrusion" based on the result of the intrusion detection; this label will
        # be used for supervised learning during training
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
        

        # Show messages that are considered intrusions (for demonstration in next meeting)
        df.select("tmst", "DevAddr", "DevEUI", "MACPayload", "MIC", "rssi", "Valid_FHDR", 
                        "Valid_MACPayload", "intrusion") \
                    .filter(df.intrusion == 1) \
                    .show(40, truncate=False)

        # apply normalization
        #df = DataPreProcessing.normalization(df)

        end_time = time.time()

        print("Time of rxpk pre-processing:", format_time(end_time - start_time), "\n\n")

        return df

