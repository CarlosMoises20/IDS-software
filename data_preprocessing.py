

from pyspark.sql.functions import expr, struct




# 1 - Converts a dataset of type 'rxpk', given the filename of the dataset, into a 'df' Spark dataframe
# 2 - Applies feature selection techniques to remove the most irrelevant attributes (dimensionality reduction),
#        selecting only the attributes that are relevant to build the intended model for IDS 
def pre_process_rxpk_dataset(df):


    ### FEATURE SELECTION: remove irrelevant / redundant attributes
    df = df.drop("type", "totalrxpk", "fromip")

    # apply filter to let pass only relevant attributes inside 'rxpk' array
    df = df.withColumn("rxpk",
                       expr("transform(rxpk, x -> named_struct( 'AppEUI', x.AppEUI, \
                            'AppNonce', x.AppNonce, \
                            'DLSettings', x.DLSettings, \
                            'DLSettingsRX1DRoffset', x.DLSettingsRX1DRoffset, \
                            'DLSettingsRX2DataRate', x.DLSettingsRX2DataRate, \
                            'DevAddr', x.DevAddr, \
                            'DevEUI', x.DevEUI, \
                            'DevNonce', x.DevNonce, \
                            'Direction', x.Direction, \
                            'FCnt', x.FCnt, \
                            'FCtrl', x.FCtrl, \
                            'FCtrlACK', x.FCtrlACK, \
                            'FCtrlADR', x.FCtrlADR, \
                            'FHDR', x.FHDR, \
                            'FOpts', x.FOpts, \
                            'FPort', x.FPort, \
                            'FRMPayload', x.FRMPayload, \
                            'MACPayload', x.MACPayload, \
                            'MHDR', x.MHDR, \
                            'MIC', x.MIC, \
                            'MessageType', x.MessageType, \
                            'NetID', x.NetID, \
                            'PHYPayload', x.PHYPayload, \
                            'RxDelay', x.RxDelay, \
                            'RxDelayDel', x.RxDelayDel, \
                            'chan', x.chan, \
                            'data', x.data, \
                            'datr', x.datr, \
                            'freq', x.freq, \
                            'lsnr', x.lsnr, \
                            'rfch', x.rfch, \
                            'rsig', transform(x.rsig, rs -> named_struct( \
                                            'chan', rs.chan, \
                                            'lsnr', rs.lsnr, \
                            )), \
                            'rssi', x.rssi, 'size', x.size, \
                            'time', x.time, 'tmms', x.tmms, 'tmst', x.tmst ))")
                    )


    ## TODO: impute missing values (using pyspark)



    return df




# 1 - Converts a dataset of type 'txpk', given the filename of the dataset, into a 'df' Spark dataframe
# 2 - Applies feature selection techniques to remove the most irrelevant attributes (dimensionality reduction),
#        selecting only the attributes that are relevant to build the intended model for IDS 
def pre_process_txpk_dataset(df):

    ## Feature Selection: remove irrelevant / redundant attributes
    df = df.drop("type")


    # apply filter to let pass only relevant attributes inside 'txpk' 
    df = df.withColumn("txpk", 
                       struct(
                           expr("txpk.data AS data"),
                           expr("txpk.datr AS datr"),
                           expr("txpk.freq AS freq"),
                           expr("txpk.powe AS powe"),
                           expr("txpk.size AS size"),
                           expr("txpk.tmst AS tmst")
                       )
                      )


    return df

