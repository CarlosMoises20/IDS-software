from pyspark.sql import SparkSession
import os
from utils import *


def process_dataset(dataset, process_type):

    # object where all the summaries of the results will be stored
    final_summary = None

    # for each file inside the directory, process the messages 
    # inside it according to the parameters on 'schema'
    for file in os.listdir(dataset):
        
        # absolute path
        filename = os.path.join(os.fsdecode(dataset), os.fsdecode(file))
        
        # Load the data from the dataset file
        df = spark.read.json(filename)

        ## the process of the file parameters depends if it has messages of type 'rxpk', 'stat' or 'txpk'
        ## so, considering each type of messages inside a file, a different function is called
        if file.decode().startswith("rxpk"):        # 'rxpk'
            summary = process_rxpk_dataset(df, process_type)
        #elif file.decode().startswith("stats"):     # 'stat'
        #    summary = process_stat_dataset(df, process_type)
        #else:                                       # 'txpk'
        #    summary = process_txpk_dataset(df, process_type)

        # Concatenate summaries
        if final_summary is None:
            final_summary = summary
        else:
            final_summary = final_summary.union(summary)

        print(f"File '{filename}' has been processed")

    return final_summary



if __name__ == '__main__':

    # Initialize Spark Session
    spark = SparkSession.builder.appName("LoRaWAN Anomaly Detection").master("local[*]").getOrCreate()


    ## Train the model

    # Output directory to store the training results
    #output_path_train = "./output_train"

    # Dataset test directory
    #dataset_train = os.fsencode('.\dataset_train')

    # Process the training dataset
    #final_summary_train = process_dataset(dataset_train, 'train')
    
    # Save the final summary in a CSV file
    #final_summary_train.write.mode("overwrite").csv(output_path_train)

    # Print the results
    #final_summary_train.show()

    # Print the name of the output directory
    #print(f"Training -> Anomaly summary saved to: {output_path_train}")


    ## Test the model

    # Output directory to store the testing results
    output_path_test = "./output_test"

    # Dataset test directory
    dataset_test = os.fsencode('.\dataset_test')

    # Process the testing dataset
    final_summary_test = process_dataset(dataset_test, 'test')

    # Save the final summary in a CSV file
    final_summary_test.write.mode("overwrite").csv(output_path_test)

    # Print the results
    final_summary_test.show()

    # Print the name of the output directory
    print(f"Testing -> Anomaly summary saved to: {output_path_test}")
    

    # Stop the Spark Session
    spark.stop()