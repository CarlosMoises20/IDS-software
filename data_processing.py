
from pyspark.sql.functions import col, when, count, explode, expr
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors        # for kNN
from sklearn import model_selection
import os
from intrusion_detection import *
from crate.client import connect
import pandas as pd
from data_pre_processing import *
from auxiliary_functions import *


### On this module, add functions, where each function process a different type of messages




def process_rxpk_dataset(spark_session, dataset):

    ### Bind all log files into a single log file if it doesn't exist yet,
    ### to simplify data processing
    
    combined_logs_filename = './combined_datasets/combined_rxpk_logs.log'
    bind_dir_files(dataset, combined_logs_filename)


    # Load the dataset into a Spark Dataframe
    df = spark_session.read.json(combined_logs_filename)


    ### Pre-Processing
    df = pre_process_rxpk_dataset(df)


    #knn = KNeighborsClassifier(n_neighbors=10, algorithm='auto')
    #knn.fit(df.rxpk.rssi, df.rxpk.snr)

    # Add anomaly detection columns
    #df = df.withColumn("Jamming", when(jamming_detection(df.rxpk.rssi), 1).otherwise(0))

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

    ### Bind all log files into a single log file if it doesn't exist yet
    ### to simplify data processing

    combined_logs_filename = './combined_datasets/combined_txpk_logs.log'
    bind_dir_files(dataset, combined_logs_filename)

    # Load the dataset into a Spark Dataframe
    df = spark_session.read.json(combined_logs_filename)

    ### Pre-Processing
    df = pre_process_txpk_dataset(df)

    pass


