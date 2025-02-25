
from pyspark.sql.types import *
from pyspark.sql.functions import expr, struct, explode, col, when
from pyspark.ml.feature import Imputer                   # includes 'mean', 'median' and 'mode'
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans



def impute_missing_values(df, df_features, replace_value):

    
    for df_feature in df_features:

        print(df_feature)

        df.select(df_feature).show()
        
        """
        df = df.withColumn(
            df_feature,
            expr(f"transform({df_feature}, x -> coalesce(x, {replace_value}))")
        )

        
        result = df_feature
        
        if isinstance(df_feature, list):
            for i in df_feature:
                print(i)
                result[i] = replace_value

        df = df.withColumn(df_feature, result)
        """
        


    return df



"""
This function applies pre-processing on data from the dataframe 'df', for the 'rxpk' dataset

 1 - Applies feature selection techniques to remove irrelevant attributes (dimensionality reduction),
        selecting only the attributes that are relevant to build the intended model_numeric for IDS 

 2 - Imputes missing values

"""
def pre_process_rxpk_dataset(df):

    ### FEATURE SELECTION: remove irrelevant / redundant attributes
    df = df.drop("type", "totalrxpk", "fromip", "ip", "port", "recv_date")


    # apply filter to let pass only relevant attributes inside 'rxpk' array and impute some missing values
    # with customized values for each attribute; 
    # for some strings, replace NULL with empty string (others are replaced with "Unknown")
    # for some numerical values, replace NULL with 0 
    #   (TODO: change others to impute MV with mean, median, kNN or Logistic Regression: 
    #               like "freq", "chan", "lsnr", "rssi")
    # for all boolean values, replace NULL with false
    df = df.withColumn("rxpk", expr("transform(rxpk, x -> named_struct( 'AppEUI', IFNULL(x.AppEUI, ''), \
                                    'AppNonce', IFNULL(x.AppNonce, ''), \
                                    'DLSettings', IFNULL(x.DLSettings, ''), \
                                    'DLSettingsRX1DRoffset', IFNULL(x.DLSettingsRX1DRoffset, ''), \
                                    'DLSettingsRX2DataRate', IFNULL(x.DLSettingsRX2DataRate, ''), \
                                    'DevAddr', IFNULL(x.DevAddr, 'Unknown'), \
                                    'DevEUI', IFNULL(x.DevEUI, 'Unknown'), \
                                    'DevNonce', IFNULL(x.DevNonce, ''), \
                                    'FCnt', IFNULL(x.FCnt, ''), \
                                    'FCtrl', IFNULL(x.FCtrl, ''), \
                                    'FCtrlACK', IFNULL(x.FCtrlACK, false), \
                                    'FCtrlADR', IFNULL(x.FCtrlADR, false), \
                                    'FHDR', IFNULL(x.FHDR, ''), \
                                    'FOpts', IFNULL(x.FOpts, ''), \
                                    'FPort', IFNULL(x.FPort, ''), \
                                    'FRMPayload', IFNULL(x.FRMPayload, ''), \
                                    'MACPayload', IFNULL(x.MACPayload, ''), \
                                    'MHDR', IFNULL(x.MHDR, ''), \
                                    'MIC', IFNULL(x.MIC, ''), \
                                    'MessageType', IFNULL(x.MessageType, ''), \
                                    'NetID', IFNULL(x.NetID, ''), \
                                    'PHYPayload', IFNULL(x.PHYPayload, ''), \
                                    'RxDelay', IFNULL(x.RxDelay, ''), \
                                    'RxDelayDel', IFNULL(x.RxDelayDel, 0), \
                                    'chan', IFNULL(x.chan, 0), \
                                    'codr', IFNULL(x.codr, ''), \
                                    'data', IFNULL(x.data, ''), \
                                    'datr', IFNULL(x.datr, ''), \
                                    'freq', x.freq, \
                                    'lsnr', IFNULL(x.lsnr, 0), \
                                    'rfch', IFNULL(x.rfch, 0), \
                                    'rsig', transform(x.rsig, rs -> named_struct( \
                                                    'chan', IFNULL(rs.chan, 0), \
                                                    'lsnr', IFNULL(rs.lsnr, 0))), \
                                    'rssi', IFNULL(x.rssi, 0), \
                                    'size', x.size, \
                                    'tmst', x.tmst ))")
                    )



    #df = impute_missing_values(df, ["rxpk.AppNonce", "rxpk.DevNonce"], 0)

    #impute_missing_values(df, [], "")
    
    
    # TODO: impute missing values for some numeric attributes: using the 'mean' strategy
    #       attributes: freq, 

    # TODO: impute missing values for the remaining string & categorical attributes: kNN / Logistic Regression
    
    
    return df




"""

This function applies pre-processing on data from the dataframe 'df_txpk', for the 'txpk' dataset

 1 - Applies feature selection techniques to remove irrelevant attributes (dimensionality reduction),
        selecting only the attributes that are relevant to build the intended model_numeric for IDS 

 2 - Imputes missing values

 (...)

"""
def pre_process_txpk_dataset(df):

    ## Feature Selection: remove irrelevant / redundant attributes
    df = df.drop("type", "recv_date", "fromip", "ip", "port", "Direction")

    # apply filter to let pass only relevant attributes inside 'txpk' 
    df = df.withColumn("txpk", struct(
                           expr("txpk.data AS data"),
                           expr("txpk.datr AS datr"),
                           expr("txpk.freq AS freq"),
                           expr("txpk.powe AS powe"),
                           expr("txpk.size AS size"),
                           expr("txpk.tmst AS tmst")
                        ))


    # TODO: impute missing values
    # for boolean attributes (FCtrlACK and FCtrlADR): replace NULL with false
    # for some numerical attributes: replace NULL with 0
    # for some string attributes: replace NULL with empty string
    df = df.na.fill({
        "AppNonce": "",
        "CFList": "",
        "DLSettings": "",
        "DLSettingsRX1DRoffset": "",
        "DLSettingsRX2DataRate": "",
        "DevAddr": "Unknown",
        "FCnt": "",
        "FCtrl": "",
        "FCtrl": "",
        "FCtrlACK": False,
        "FCtrlADR": False,
        "FHDR": "",
        "FCnt": "",
        "FOpts": "",
        "FPort": "",
        "FRMPayload": "",
        "FreqCh4": "",
        "FreqCh5": "",
        "FreqCh6": "",
        "FreqCh7": "",
        "FreqCh8": "",
        "MACPayload": "",
        "MHDR": "",
        "MIC": "",  
        "MessageType": "",
        "NetID": "",
        "PHYPayload": "",
        "RxDelay": "",
        "RxDelayDel": 0,
        #"txpk.data": ""    # This is not supported for nested...
        #"txpk.datr": ""    # This is not supported for nested...
    })


    # TODO: "freq" (and, eventually, other numeric attributes that 
    # were not imputed with 0): replace NULL with mean
    #mv_txpk_imputer = Imputer(inputCol="df.txpk.freq", outputCol="df.txpk.freq", strategy="mean")

    #mv_txpk_imputer_model = mv_txpk_imputer.fit(df)

    #df = mv_txpk_imputer_model.transform(df)

    return df

