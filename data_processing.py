
from pyspark.sql.functions import col, when, count, explode, expr
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors        # for kNN
from sklearn import model_selection
import os
from intrusion_detection import *
from crate.client import connect
import pandas as pd


### On this module, add functions, where each function process a different type of messages


# Auxiliary function to bind all log files inside a indicated directory
# into a single log file of each type of LoRaWAN message
def bind_dir_files(dataset_path, output_filename):

    all_logs = []                         # Create a list to store the different files

    for filename in dataset_path:
        with open(filename, 'r') as f:
            all_logs.append(f.read())     # Append the contents of the file to the list


    # Join all logs into a single string
    combined_logs = '\n'.join(all_logs)

    # Write the combined logs to a new file
    with open(output_filename, 'w') as f:
        f.write(combined_logs)



# 1 - Converts a dataset of type 'rxpk', given the filename of the dataset, into a 'df' Spark dataframe
# 2 - Applies feature selection techniques to remove the most irrelevant attributes (dimensionality reduction),
#        selecting only the attributes that are relevant to build the intended model for IDS 
def pre_process_rxpk_dataset(df):


    ## Feature Selection: remove irrelevant / redundant attributes
    df = df.drop("type", "totalrxpk", "fromip")

    df = df.withColumn("rxpk", \
                       expr("transform(rxpk, x -> named_struct( 'AppEUI', x.AppEUI, \
                            'AppNonce', x.AppNonce, 'DLSettings', x.DLSettings, \
                            'DLSettingsRX1DRoffset', x.DLSettingsRX1DRoffset, \
                            'DLSettingsRX2DataRate', x.DLSettingsRX2DataRate, \
                            'DevAddr', x.DevAddr, 'DevEUI', x.DevEUI, 'DevNonce', x.DevNonce, \
                            'Direction', x.Direction, 'FCnt', x.FCnt, \
                            'FCtrl', x.FCtrl, 'FCtrlACK', x.FCtrlACK, \
                            'FCtrlADR', x.FCtrlADR, 'FHDR', x.FHDR, \
                            'FOpts', x.FOpts, 'FPort', x.FPort, \
                            'FRMPayload', x.FRMPayload,'MACPayload', x.MACPayload, \
                            'MHDR', x.MHDR, 'MIC', x.MIC, \
                            'MessageType', x.MessageType, 'NetID', x.NetID, \
                            'PHYPayload', x.PHYPayload, 'RxDelay', x.RxDelay, \
                            'RxDelayDel', x.RxDelayDel, 'aesk', x.aesk, \
                            'brd', x.brd, 'chan', x.chan, \
                            'codr', x.codr, 'data', x.data, 'datr', x.datr, \
                            'freq', x.freq, 'jver', x.jver, \
                            'lsnr', x.lsnr, 'rfch', x.rfch, \
                            'rsig', transform(x.rsig, rs -> named_struct( \
                                            'ant', rs.ant, \
                                            'chan', rs.chan, \
                                            'dbg1', rs.dbg1, \
                                            'dbg2', rs.dbg2, \
                                            'etime', rs.etime, \
                                            'foff', rs.foff, \
                                            'ftdelta', rs.ftdelta, \
                                            'ftime', rs.ftime, \
                                            'ftstat', rs.ftstat, \
                                            'ftver', rs.ftver, \
                                            'lsnr', rs.lsnr, \
                                            'rssic', rs.rssic, \
                                            'rssis', rs.rssis, \
                                            'rssisd', rs.rssisd \
                            )), \
                            'rssi', x.rssi, 'size', x.size, \
                            'time', x.time, 'tmms', x.tmms, 'tmst', x.tmst ))")
                    )

    return df




# 1 - Converts a dataset of type 'txpk', given the filename of the dataset, into a 'df' Spark dataframe
# 2 - Applies feature selection techniques to remove the most irrelevant attributes (dimensionality reduction),
#        selecting only the attributes that are relevant to build the intended model for IDS 
def pre_process_txpk_dataset(df):

    ## Feature Selection: remove irrelevant / redundant attributes
    df = df.drop("type")

    return df




def process_rxpk_dataset(spark_session, dataset):

    ### Bind all log files into a single log file
    
    combined_logs_filename = './combined_datasets/combined_rxpk_logs.log'
    bind_dir_files(dataset, combined_logs_filename)


    # Load the dataset into a Spark Dataframe

    df = spark_session.read.json(combined_logs_filename)


    ### Pre-Processing
    
    df = pre_process_rxpk_dataset(df)


    #knn = KNeighborsClassifier(n_neighbors=10, algorithm='auto')
    #knn.fit(df.rxpk.rssi, df.rxpk.snr)

    # Add anomaly detection columns
    df = df.withColumn("Jamming", when(jamming_detection(df.rxpk.rssi), 1).otherwise(0))

    # Combine anomaly indicators (TODO: juntar com todas as outras anomalias)
    #df = df.withColumn("Anomaly", (col("Jamming")) > 0)

    # Group data for analysis
    """
    summary = df.groupBy("Anomaly").agg(
        count("*").alias("MessageCount"),
        count(when(col("Anomaly_SNR") == 1, 1)).alias("SNR_Anomalies"),
        count(when(col("Anomaly_RSSI") == 1, 1)).alias("RSSI_Anomalies"),
        count(when(col("Anomaly_MIC") == 1, 1)).alias("MIC_Anomalies"),
        count(when(col("Anomaly_Size") == 1, 1)).alias("Size_Anomalies")
    )
    """


    pass




def process_txpk_dataset(spark_session, dataset):

    ### Bind all log files into a single log file

    combined_logs_filename = './combined_datasets/combined_txpk_logs.log'
    bind_dir_files(dataset, combined_logs_filename)

    # Load the dataset into a Spark Dataframe

    df = spark_session.read.json(combined_logs_filename)

    ### Pre-Processing
    
    df = pre_process_txpk_dataset(df)

    pass





"""
vectorized_data = [.....]

# Create a kNN model
model = KNeighborsClassifier(n_neighbors=10, algorithm='auto')

model.fit(vectorized_data)

model.predict(vectorized_data)

"""