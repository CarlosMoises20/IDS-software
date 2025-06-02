
import time
from common.auxiliary_functions import format_time
from common.spark_functions import get_boolean_attributes_names, train_test_split
from pyspark.sql.functions import when, col, lit, expr, length, regexp_extract, udf, sum
from pyspark.sql.types import FloatType, IntegerType
from prepareData.preProcessing.pre_processing import DataPreProcessing
from pyspark.ml.feature import Imputer


"""
Applies pre-processing steps specified for each dataset type ("rxpk" and "txpk")
Binds the results and prepares the whole dataframe with pre-processing steps common for all
"rxpk" and "txpk" samples of the entire dataframe

"""
def pre_process_type(df, dataset_type):

    # separate 'rxpk' samples or 'txpk' samples to apply pre-processing steps that are
    # specific to each type of LoRaWAN message
    df = dataset_type.value["pre_processing_class"].pre_process_data(df)


    # Replace NULL and empty-string values of DevAddr with "Unknown"
    # this opens possibilities to detect attacks of devices that didn't join the network probably because they were
    # targeted by some sort of attack in the LoRaWAN physical layer
    df = df.withColumn("DevAddr", when((col("DevAddr").isNull()) | (col("DevAddr") == lit("")), "Unknown")
                                    .otherwise(col("DevAddr")))
    

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
    hex_attributes = ["AppNonce", "CFListType",
                        "FCnt", "FCtrl", "FOpts", "FPort", 
                        "MIC", "NetID", "RxDelay"]

    df = DataPreProcessing.hex_to_decimal(df, hex_attributes)

    # add the label with the value 0 (that will be later updated)
    return df.withColumn("intrusion", lit(0))



"""
Applies pre-processing steps for all "rxpk" and "txpk" rows of the dataframe for
a specific device, if specified

"""
def prepare_df_for_device(spark_session, dataset_type, dev_addr):

    start_time = time.time()

    df_model_train = spark_session.read.json(
        f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_train.json'
    ).filter(col("DevAddr") == dev_addr)

    df_model_test = spark_session.read.json(
        f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_test.json'
    ).filter(col("DevAddr") == dev_addr)

    df_model_train = df_model_train.cache()
    df_model_test = df_model_test.cache()

    df_model_train_count = df_model_train.count()
    df_model_test_count = df_model_test.count()

    # NOTE: uncomment these two lines to print the number of training and testing samples for the device
    #print(f'Number of {dataset_type.value["name"].upper()} training samples for device {dev_addr}: {df_model_train_count}')
    #print(f'Number of {dataset_type.value["name"].upper()} testing samples for device {dev_addr}: {df_model_test_count}')

    # If there are no samples for the device, it's not possible to create a model since there is
    # no data to be used to train the model
    if df_model_train_count == 0 and df_model_test_count == 0:
        print(f'There are no samples for the device {dev_addr} for {dataset_type.value["name"].upper()}. No model will be created.\n\n\n')
        return None, None

    # If there are samples for the device but there is imbalance in the number of training and testing samples,
    # it's necessary to apply a new division of all the samples in training and testing in order to have sufficient
    # samples specially for training
    if df_model_train_count == 0 or df_model_test_count == 0 or df_model_train_count * 0.85 < df_model_test_count:
        
        print(f"[INFO] Adjusting sample split due to imbalance...")

        # Binds training and testing datasets
        df_model = df_model_train.unionByName(df_model_test, allowMissingColumns=True)

        # Applies new division in training and testing based in dataset size rules
        df_model_train, df_model_test = train_test_split(df_model)

    # NOTE: uncomment these two lines to print the number of training and testing samples for the device after sample redistribution
    #print(f'Number of {dataset_type.value["name"].upper()} training samples for device {dev_addr} after sample redistribution: {df_model_train.count()}')
    #print(f'Number of {dataset_type.value["name"].upper()} testing samples for device {dev_addr} after sample redistribution: {df_model_test.count()}')

    # Remove columns where all values are null
    non_null_columns_train = [
        c for c in df_model_train.columns
        if (df_model_train.agg(sum(when(col(c).isNotNull(), 1).otherwise(0))).first()[0] or 0) > 0
    ]

    non_null_columns_test = [
        c for c in df_model_test.columns
        if (df_model_test.agg(sum(when(col(c).isNotNull(), 1).otherwise(0))).first()[0] or 0) > 0
    ]

    df_model_train = df_model_train.select(non_null_columns_train)
    df_model_test = df_model_test.select(non_null_columns_test)

    non_null_columns_train = list(set(non_null_columns_train) - set(["DevAddr", "intrusion"]))
    non_null_columns_test = list(set(non_null_columns_test) - set(["DevAddr", "intrusion"]))
    
    # replace NULL and empty values with the mean on numeric attributes with missing values, because these are values
    # that can assume any numeric value, so it's not a good approach to replace missing values with a static value
    # the mean is the best approach to preserve the distribution and variety of the data
    imputer_train = Imputer(inputCols=non_null_columns_train, outputCols=non_null_columns_train, strategy="mean")
    imputer_test = Imputer(inputCols=non_null_columns_test, outputCols=non_null_columns_test, strategy="mean")

    df_model_train, df_model_test = imputer_train.fit(df_model_train).transform(df_model_train), imputer_test.fit(df_model_test).transform(df_model_test)

    # apply normalization
    df_model_train, df_model_test = DataPreProcessing.normalization(df_model_train), DataPreProcessing.normalization(df_model_test)

    end_time = time.time()

    # Print the total time of pre-processing
    print(f'Total time of pre-processing in device {dev_addr} and {dataset_type.value["name"].upper()}: {format_time(end_time - start_time)} \n')

    return df_model_train, df_model_test

