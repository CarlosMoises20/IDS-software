from pyspark.sql import SparkSession
import os
from data_processing import *


def process_dataset(spark_session, dataset):

    # object where all the summaries of the results will be stored
    final_summary_txpk = None
    final_summary_rxpk = None


    # TODO: later, implement 2 spark jobs to execute tasks in parallel: one for 'rxpk' and one for 'txpk'
    # by now, we are doing everything locally

    ### Uplink Messages

    dataset_txpk = [os.path.join(os.fsdecode(dataset), os.fsdecode(file))
                   for file in os.listdir(dataset) if file.decode().startswith("txpk")]


    final_summary_txpk = process_txpk_dataset(spark_session, dataset_txpk)


    ### Downlink Messages

    dataset_rxpk = [os.path.join(os.fsdecode(dataset), os.fsdecode(file)) 
                  for file in os.listdir(dataset) if file.decode().startswith("rxpk")]

    final_summary_rxpk = process_rxpk_dataset(spark_session, dataset_rxpk)


    return final_summary_txpk, final_summary_rxpk



if __name__ == '__main__':

    # Initialize Spark Session
    spark_session = SparkSession.builder.appName("IDS for LoRaWAN network").master("local[*]").getOrCreate()

    dataset_path = "./dataset"

    final_summary_test_uplink, final_summary_test_downlink = process_dataset(spark_session, os.fsencode(dataset_path))
    

    # Print the test result of uplink messages in the console

    print("Uplink messages: \n")
    print(final_summary_test_uplink)


    # Print the test result of downlink messages in the console

    print("\n\nDownlink messages: \n")
    print(final_summary_test_downlink)


    # TODO: save the result of the tests in two unique CSV files instead of one file for each processed file

    final_summary_test_uplink.write.mode("overwrite").csv("./output_test_uplink")
    final_summary_test_downlink.write.mode("overwrite").csv("./output_test_downlink")



    # Stop the Spark Session
    spark_session.stop()