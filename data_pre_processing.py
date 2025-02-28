
from pyspark.sql.types import *
from pyspark.sql.functions import expr, struct, col, explode, udf
from auxiliary_functions import hex_to_decimal



"""
This function applies pre-processing on data from the dataframe 'df', for the 'rxpk' dataset

 1 - Applies feature selection techniques to remove irrelevant attributes (dimensionality reduction),
        selecting only the attributes that are relevant to build the intended model_numeric for IDS 


"""
def pre_process_rxpk_dataset(df):

    ## Feature Selection: remove irrelevant, redundant and correlated attributes
    df = df.drop("type", "totalrxpk", "fromip", 
                 "ip", "port", "recv_date")



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
                                    'FCtrlACK', CASE 
                                                    WHEN x.FCtrlACK IS NOT NULL AND x.FCtrlACK = true THEN 1 
                                                    WHEN x.FCtrlACK IS NOT NULL AND x.FCtrlACK = false THEN 0 
                                                    ELSE NULL 
                                                END,
                                    'FCtrlADR', CASE 
                                                    WHEN x.FCtrlADR IS NOT NULL AND x.FCtrlADR = true THEN 1 
                                                    WHEN x.FCtrlADR IS NOT NULL AND x.FCtrlADR = false THEN 0 
                                                    ELSE NULL 
                                                END,
                                    'FHDR', x.FHDR, 
                                    'FOpts', x.FOpts, 
                                    'FPort', x.FPort, 
                                    'FRMPayload', x.FRMPayload, 
                                    'MACPayload', x.MACPayload, 
                                    'MHDR', x.MHDR, 
                                    'MIC', x.MIC, 
                                    'MessageType', x.MessageType, 
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
    
    # TODO: convert some hexadecimal attributes to decimal or binary
    
    # explode 'rxpk' array, since each element of the array corresponds to a different LoRaWAN message
    df = df.withColumn("rxpk", explode(col("rxpk")))

    # change attribute names to be recognized by PySpark (for example, 'rxpk.AppEUI' -> 'AppEUI')
    df = df.select("rxpk.*")

    # TODO: after starting processing, analyse if it's necessary to apply some more pre-processing steps

    return df




"""

This function applies pre-processing on data from the dataframe 'df_txpk', for the 'txpk' dataset

 1 - Applies feature selection techniques to remove irrelevant attributes (dimensionality reduction),
        selecting only the attributes that are relevant to build the intended model_numeric for IDS 


"""
def pre_process_txpk_dataset(df):

    ## Feature Selection: remove irrelevant, redundant and correlated attributes
    df = df.drop("type", "recv_date", "fromip", "ip", "port", 
                 "Direction", "RxDelayDel", "DLSettings")

    # apply filter to let pass only relevant attributes inside 'txpk' 
    df = df.withColumn("txpk", struct(
                           expr("txpk.data AS data"),
                           expr("txpk.datr AS datr"),
                           expr("txpk.freq AS freq"),
                           expr("txpk.powe AS powe"),
                           expr("txpk.size AS size"),
                           expr("txpk.tmst AS tmst")
                        ))


    # create a new attribute called "CFListType", coming from the last octet of "CFList" according to the LoRaWAN v1.1 specification
    # source: https://lora-alliance.org/resource_hub/lorawan-specification-v1-1/ 
    df = df.withColumn("CFListType", expr("substring(CFList, -2, 2)"))

    # remove the "CFList" attribute, since it's already split to "FreqCh4", "FreqCh5", "FreqCh6", 
    # "FreqCh7", "FreqCh8" and "CFListType", for a more simple processing
    df = df.drop("CFList")


    # TODO: after starting processing, analyse if it's necessary to apply some more pre-processing steps

    return df

