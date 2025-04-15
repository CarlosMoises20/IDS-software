
import time
from common.auxiliary_functions import bind_dir_files, get_all_attributes_names, get_boolean_attributes_names, format_time
from pyspark.sql.functions import when, col, lit, expr, length, regexp_extract, udf
from pyspark.sql.types import FloatType, IntegerType
from prepareData.preProcessing.pre_processing import DataPreProcessing
from common.dataset_type import DatasetType
from common.constants import SPARK_NUM_PARTITIONS
from pyspark.ml.feature import Imputer


#### Pre-processing steps specified for the dataset type
def pre_process_type(spark_session, dataset_type):

    ### Bind all log files into a single log file if it doesn't exist yet,
    ### to simplify data processing
    combined_logs_filename = bind_dir_files(spark_session, dataset_type)

    # Load the parquet dataset into a Spark Dataframe
    df = spark_session.read.parquet(combined_logs_filename)

    ### Apply pre-processing techniques only specified for the dataset type
    return dataset_type.value["pre_processing_class"].pre_process_data(df)


#### Pre-processing steps common for both dataset types
def pre_process_general(df):

    ### Apply transformations to attributes

    # create a new attribute called "CFListType", coming from the last octet of "CFList" according to the LoRaWAN specification
    # source: https://lora-alliance.org/resource_hub/lorawan-specification-v1-1/ (or specification of any other LoRaWAN version than v1.1)
    df = df.withColumn("CFListType", when((col("CFList").isNull()) | (col("CFList") == lit("")), None)
                                        .otherwise(expr("substring(CFList, -2, 2)")))

    # remove the "CFList" attribute, since it's already split to "FreqCh4", "FreqCh5", "FreqCh6", 
    # "FreqCh7", "FreqCh8" (on TXPK; in RXPK it's not necessary) and "CFListType" (in both RXPK and TXPK), 
    # for a more simple processing
    df = df.drop("CFList")

    # Create 'dataLen' and 'FRMPayload_Len' attributes that correspond to the length of 'data' and 'FRMPayload', 
    # that represents the content of the LoRaWAN message; we only need their lengths to detect anomalies in the data size
    # and the ML algorithms only work with numerical features so they wouldn't read the data as values, but as categories,
    # which is not supposed
    df = df.withColumn("dataLen", length(col("data"))) \
            .withColumn("FRMPayload_Len", length(col("FRMPayload")))
    
    # remove 'data' and 'FRMPayload' after computing their lengths
    df = df.drop("data", "FRMPayload")

    # Replace NULL and empty-string values of DevAddr with "Unknown"
    # this opens possibilities to detect attacks of devices that didn't join the network probably because they were
    # targeted by some sort of attack in the LoRaWAN physical layer
    df = df.withColumn("DevAddr", when((col("DevAddr").isNull()) | (col("DevAddr") == lit("")), "Unknown")
                                    .otherwise(col("DevAddr")))
    
    # Convert "codr" from string to float
    str_float_udf = udf(DataPreProcessing.str_to_float, FloatType())
    df = df.withColumn("codr", str_float_udf(col("codr")))

    ### Extract values of LoRa parameters SF and BW from "datr" attribute
    pattern = r"SF(\d+)BW(\d+)"     # regex pattern to extract "SF" and "BW" 

    df = df.withColumn("SF", regexp_extract(col("datr"), pattern, 1).cast(IntegerType())) \
            .withColumn("BW", regexp_extract(col("datr"), pattern, 2).cast(IntegerType()))

    # Remove "datr" after splitting it by "SF" and "BW"
    df = df.drop("datr")


    ### Ensure all attributes are in integer format

    # Boolean attributes to integer (True -> 1; False -> 0; NULL -> -1)
    boolean_attributes = get_boolean_attributes_names(df.schema)

    df = DataPreProcessing.bool_to_int(df, boolean_attributes)

    # hexadecimal attributes to decimal common for all dataset types
    # this also replaces NULL and empty values with -1 to be supported by the algorithms
    # if we want to apply machine learning algorithms, we need numerical values and if these values stayed as strings,
    # these would be treated as categorical values, which is not the case
    hex_attributes = ["AppNonce", "FreqCh4", "FreqCh5", 
                        "FreqCh6", "FreqCh7", "FreqCh8",
                        "CFListType", "DevEUI", "DevNonce",
                        "FCnt", "FCtrl", "FOpts", "FPort", 
                        "MIC", "NetID", "RxDelay"]
    
    df = DataPreProcessing.hex_to_decimal(df, hex_attributes)

    # get all other attributes of the dataframe
    remaining_attributes = list(set(get_all_attributes_names(df.schema)) - 
                                set(hex_attributes + boolean_attributes + ["DevAddr"]))
    
    # for the other numeric attributes, replace NULL and empty values with the mean, because these are values
    # that can assume any numeric value, so it's not a good approach to replace missing values with a static value
    # the mean is the best approach to preserve the distribution and variety of the data
    imputer = Imputer(inputCols=remaining_attributes, outputCols=remaining_attributes, strategy="mean")

    df = imputer.fit(df).transform(df)

    # apply normalization
    df = DataPreProcessing.normalization(df)

    return df


"""
This function is called to apply all necessary pre-processing steps to prepare
the dataset to be processed by the used ML models on the IDS

It extracts all "rxpk" and "txpk" LoRaWAN messages from the given datasets, that are
in JSON format, and converts them to a spark dataframe

For that, it receives:

    spark_session: the Spark session used to read all messages in a file and 
        convert them into a spark dataframe

    dataset_types: the types of dataset ("rxpk" and "txpk")

"""
def prepare_past_dataset(spark_session):

    start_time = time.time()

    df_rxpk, df_txpk = (pre_process_type(spark_session, dataset_type) 
                            for dataset_type in [key for key in DatasetType])

    # after pre-processing, combine 'rxpk' and 'txpk' dataframes in just one  
    df = df_rxpk.unionByName(df_txpk, allowMissingColumns=True)

    df = df.coalesce(numPartitions=int(SPARK_NUM_PARTITIONS))

    # Caching saves time by retrieving data from memory instead of always retrieving from the sources,
    # especially in repeated computations
    df.cache()

    # apply pre-processing techniques common to "rxpk" and "txpk"
    df = pre_process_general(df)

    end_time = time.time()

    # Print the total time of pre-processing; the time is in seconds, minutes or hours
    print("Total time of pre-processing:", format_time(end_time - start_time), "\n\n")

    return df