
import time
from preProcessing.pre_processing import DataPreProcessing
from pyspark.sql.functions import expr, col, explode, length, when, col, udf, concat, asc, desc, struct
from pyspark.sql.types import StringType, IntegerType, BooleanType, DoubleType, NumericType
from pyspark.ml.feature import Imputer
from auxiliaryFunctions.general import get_all_attributes_names, format_time
from auxiliaryFunctions.anomaly_detection import AnomalyDetection
from constants import *


"""
This class represents the pre-processing phase on LoRaWAN messages from the 'rxpk' dataset

"""
class RxpkPreProcessing(DataPreProcessing):

    # this function will be fundamental to define the label that will be used as
    # the desired output in the model training, that will allow the model to fit
    # its weights and pendors using the input and the desired output
    @staticmethod
    def intrusion_detection(df_row):

        try:

            # TODO: introduce all types of possible LoRaWAN attacks here
            # TODO: correct the logic; separate examples for each device (DevAddr)

            # call AnomalyDetection.__detection

            # Example logic for detecting an intrusion
            jamming = AnomalyDetection.jamming_detection(df_row.rssi)

            # Example LSNR anomaly detection (adjust thresholds as needed)
            lsnr_anomaly = df_row.lsnr1 < LSNR_MIN or df_row.lsnr2 < LSNR_MIN or \
                            df_row.lsnr2 > LSNR_MAX or df_row.lsnr2 > LSNR_MAX

            #replay_attack = AnomalyDetection.__replay_attack(fcnt_history, df_row.FCnt)
            
            #sinkhole = AnomalyDetection.sinkhole_detection(df_row.freq)
            #wormhole = AnomalyDetection.wormhole_detection(df_row.tmst)
            downlink_routing = AnomalyDetection.downlink_routing_attack(df_row.Valid_MACPayload)
            physical_tampering = AnomalyDetection.physical_tampering(df_row.Valid_FHDR)


            # If any of the conditions indicate an intrusion, return 1 (intrusion detected), otherwise return 0
            return int(jamming or lsnr_anomaly or downlink_routing or physical_tampering)

        except Exception as e:
            print("Error:", e)



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
        
        # Create a udf to compare fields that correspond to part of 
        # others but with reversed octets
        reverse_hex = udf(DataPreProcessing.reverse_hex_octets, StringType())

        # TODO: analyse if these steps are necessary
        df = df.withColumn("Valid_FHDR", when(col("FHDR").isNull(), -1)
                                          .when(col("FHDR") == concat(reverse_hex(col("DevAddr")), 
                                                                        reverse_hex(col("FCtrl")), 
                                                                        reverse_hex(col("FCnt")), 
                                                                        col("FOpts")), 1)
                                           .otherwise(0))

        df = df.withColumn("Valid_MACPayload", when(col("MACPayload").isNull(), -1)
                                                .when(col("MACPayload") == concat(col("FHDR"), 
                                                                            col("FPort"), 
                                                                            col("FRMPayload")), 1)
                                                .otherwise(0))

        
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

        # Convert hexadecimal attributes (string) to numeric (DecimalType), replacing NULL and empty values with -1 since
        # -1 would never be a valid value for an hexadecimal-to-decimal attribute
        df = DataPreProcessing.hex_to_decimal(df, hex_attributes)

        # get all non-hexadecimal attributes of the dataframe
        non_hex_attributes = list(set(get_all_attributes_names(df.schema)) - set(hex_attributes))
        
        # for the other numeric attributes, replace NULL and empty values with the mean, because these are values
        # that can assume any numeric value, so it's not a good approach to replace missing values with a static value
        # the mean is the best approach to preserve the distribution and variety of the data
        imputer = Imputer(inputCols=non_hex_attributes, outputCols=non_hex_attributes, strategy="mean")

        df = imputer.fit(df).transform(df)

        # define the label "intrusion" based on the result of the intrusion detection; this label will
        # be used for supervised learning of the models during training
        # Define "intrusion" based on MessageType without using UDFs
        df = df.withColumn("intrusion", when(col("MessageType") == 0 | col("MessageType") == 6,  # Join Request and Rejoin-Request
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
                                        )       # Rejoin-Requests (6) don't exist on the dataset, so the model won't be trained on it
                                        .when(
                                            col("MessageType") == 7,  # Proprietary
                                            when(col("Valid_FHDR") == 0, 1)
                                            .when(col("Valid_MACPayload") == 0, 1)
                                            .otherwise(0)
                                        ).otherwise(0)  # No MessageType, no intrusion
                                    )

        # apply normalization
        #df = DataPreProcessing.normalization(df)

        end_time = time.time()

        print("Time of rxpk pre-processing:", format_time(end_time - start_time), "\n\n")

        return df

