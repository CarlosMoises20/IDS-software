
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


    # apply filter to let pass only relevant attributes inside 'rxpk' array
    df = df.withColumn("rxpk", expr("transform(rxpk, x -> named_struct( 'AppEUI', x.AppEUI, \
                                    'AppNonce', x.AppNonce, \
                                    'DLSettings', x.DLSettings, \
                                    'DLSettingsRX1DRoffset', x.DLSettingsRX1DRoffset, \
                                    'DLSettingsRX2DataRate', x.DLSettingsRX2DataRate, \
                                    'DevAddr', x.DevAddr, \
                                    'DevEUI', x.DevEUI, \
                                    'DevNonce', x.DevNonce, \
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
                                    'codr', x.codr, \
                                    'data', x.data, \
                                    'datr', x.datr, \
                                    'freq', x.freq, \
                                    'lsnr', x.lsnr, \
                                    'rfch', x.rfch, \
                                    'rsig', transform(x.rsig, rs -> named_struct( \
                                                    'chan', rs.chan, \
                                                    'lsnr', rs.lsnr)), \
                                    'rssi', x.rssi, \
                                    'size', x.size, \
                                    'tmst', x.tmst ))")
                    )



    
    # TODO: impute missing values for some numeric attributes: using the 'mean' strategy
    #       attributes: rssi

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


    # TODO: impute missing values (use Random Forest)

    # TODO: other numeric attributes that 
    # were not imputed with 0: replace NULL with mean
    #mv_txpk_imputer = Imputer(inputCol="txpk", outputCol="txpk", strategy="mean")

    #mv_txpk_imputer_model = mv_txpk_imputer.fit(df)

    #df = mv_txpk_imputer_model.transform(df)

    return df

