
from preProcessing.pre_processing import DataPreProcessing
from pyspark.sql.functions import expr, col, explode, length


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
                                        'MType', CASE 
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
        

        # explode 'rxpk' array, since each element inside the 'rxpk' array corresponds to a different LoRaWAN message
        df = df.withColumn("rxpk", explode(col("rxpk")))

        # change attribute names to be recognized by PySpark ML algorithms (for example, 'rxpk.AppEUI' -> 'AppEUI')
        # this also removes all attributes outside the 'rxpk' array, since these are irrelevant / redundant
        df = df.select("rxpk.*")    

        # TODO: maybe remove 'rsig'
        
        # TODO: calculate RFU


        # After calculating RFU, remove DLSettings
        df = df.drop("DLSettings")
        
        # Create 'dataLen' that corresponds to the length of 'data', 
        # that represents the content of the LoRaWAN message
        df = df.withColumn("dataLen", length(col("txpk.data")))

        # Convert hexadecimal attributes (string) to decimal (int)
        df = DataPreProcessing.hex_to_decimal(df, ["PHYPayload", "MHDR", "MACPayload",
                                                   "MIC", "FHDR", "FPort", "FRMPayload",
                                                   "DevAddr", "FCnt", "FCtrl", "FOpts",
                                                   "AppNonce", "NetID"])

        # TODO: after starting processing, analyse if it's necessary to apply some more pre-processing steps

        return df

