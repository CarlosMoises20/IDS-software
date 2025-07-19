
import time
from common.auxiliary_functions import format_time
from common.spark_functions import get_boolean_attributes_names, train_test_split
from pyspark.sql.functions import when, col, lit, expr, length, regexp_extract, udf, sum
from pyspark.sql.types import FloatType, IntegerType, StringType, StructType, StructField
from common.constants import SF_LIST, BW_LIST, DATA_LEN_LIST_ABNORMAL
from prepareData.preProcessing.pre_processing import DataPreProcessing
from common.spark_functions import modify_device_dataset
from pyspark.ml.feature import Imputer


"""
Applies pre-processing steps specified for each dataset type ("rxpk" and "txpk")
Binds the results and prepares the whole dataframe with pre-processing steps common for all
"rxpk" and "txpk" samples of the entire dataframe

"""
def pre_process_type(df, dataset_type, streamProcessing=False):

    # separate 'rxpk' or 'txpk' samples to apply pre-processing steps that are
    # specific to each type of LoRaWAN message
    df = dataset_type.value["pre_processing_class"].pre_process_data(df, streamProcessing)

    if streamProcessing:

        parse_phy_payload = udf(DataPreProcessing.parse_data, StructType([
            StructField("AppEUI", StringType(), True),
            StructField("AppNonce", StringType(), True),
            StructField("DLSettings", StringType(), True),
            StructField("CFList", StringType(), True),
            StructField("CFListType", StringType(), True),
            StructField("DevAddr", StringType(), True),
            StructField("DevEUI", StringType(), True),
            StructField("DevNonce", StringType(), True),
            StructField("FCnt", StringType(), True),
            StructField("FCtrl", StringType(), True),
            StructField("FOpts", StringType(), True),
            StructField("FPort", StringType(), True),
            StructField("PHYPayloadLen", IntegerType(), True),
            StructField("MIC", StringType(), True),
            StructField("MHDR", StringType(), True),
            StructField("RxDelay", StringType(), True),
        ]))

        df = df.withColumn("parsed", parse_phy_payload(col("data")))

        # the following fields are derived from PHYPayload, if they are NULL, try to compute them directly from PHYPayload
        df = df.withColumn("AppEUI", col("parsed.AppEUI")) \
               .withColumn("AppNonce", col("parsed.AppNonce")) \
               .withColumn("DLSettings", col("parsed.DLSettings")) \
               .withColumn("CFList", col("parsed.CFList")) \
               .withColumn("CFListType", col("parsed.CFListType")) \
               .withColumn("DevAddr", col("parsed.DevAddr")) \
               .withColumn("DevEUI", col("parsed.DevEUI")) \
               .withColumn("DevNonce", col("parsed.DevNonce")) \
               .withColumn("FCnt", col("parsed.FCnt")) \
               .withColumn("FCtrl", col("parsed.FCtrl")) \
               .withColumn("FOpts", col("parsed.FOpts")) \
               .withColumn("FPort", col("parsed.FPort")) \
               .withColumn("PHYPayloadLen", col("parsed.PHYPayloadLen")) \
               .withColumn("MIC", col("parsed.MIC")) \
               .withColumn("MHDR", col("parsed.MHDR")) \
               .withColumn("RxDelay", col("parsed.RxDelay"))

        df = df.drop("parsed", "data")

        df = DataPreProcessing.hex_to_decimal(df, ["AppEUI", "DevEUI", "DevNonce"])

    else:
            # Create 'PHYPayloadLen' attributes that correspond to the length of 'PHYPayload', 
            # that represents the full content of the LoRaWAN message physical payload; we only need the length to detect anomalies
            # because we already have as attributes the results from the division of this attribute, which is too large to be processed by spark
            # and the ML algorithms only work with numerical features so they wouldn't read the data as values, but as categories,
            # which is not supposed
            df = df.withColumn("PHYPayloadLen", length(col("PHYPayload")))

            # remove 'PHYPayload' after computing its length
            df = df.drop("PHYPayload")

    # Remove samples that do not contain a DevAddr, since these samples are messages from devices that didn't join
    # the network, and having a model to learn traffic from several devices without DevAddr attributed by the network
    # does not make sense, because we want a model for each device to learn the traffic behaviour from that device only
    df = df.filter((col("DevAddr").isNotNull()) & (col("DevAddr") != ""))
    
    # create a new attribute called "CFListType", coming from the last octet of "CFList" according to the LoRaWAN specification
    # source: https://lora-alliance.org/resource_hub/lorawan-specification-v1-1/ (or specification of any other LoRaWAN version than v1.1)
    df = df.withColumn("CFListType", when(((col("CFList").isNull()) | (col("CFList") == lit(""))), None)
                                    .otherwise(expr("substring(CFList, -2, 2)")))

    df = df.withColumn("CFList", when(((col("CFList").isNull()) | (col("CFList") == lit(""))), None)
                                    .otherwise(expr("substring(CFList, 1, length(CFList) - 2)")))

    
    # Convert "codr" from string to float
    str_float_udf = udf(DataPreProcessing.str_to_float, FloatType())
    df = df.withColumn("codr", str_float_udf(col("codr")))

    ### Extract values of LoRa parameters SF and BW from "datr" attribute
    pattern = r"SF(\d+)BW(\d+)"     # regex pattern to extract "SF" and "BW" 

    # If pattern is not found, it returns -1 instead of throwing an error
    df = df.withColumn("SF", when(length(regexp_extract(col("datr"), pattern, 1)) == 0, -1)  # if empty string, put -1
                            .otherwise(regexp_extract(col("datr"), pattern, 1).cast(IntegerType()))
                        ).withColumn("BW", when(length(regexp_extract(col("datr"), pattern, 2)) == 0, -1)
                                        .otherwise(regexp_extract(col("datr"), pattern, 2).cast(IntegerType()))
                                    )

    # Remove "datr" after splitting it by "SF" and "BW"
    df = df.drop("datr")

    ### Ensure all attributes are in integer format

    # Boolean attributes to integer (True -> 1; False -> 0)
    boolean_attributes = get_boolean_attributes_names(df.schema)
    df = DataPreProcessing.bool_to_int(df, boolean_attributes)

    # hexadecimal attributes to decimal common for all dataset types
    # if we want to apply machine learning algorithms, we need numerical values and if these values stayed as strings,
    # these would be treated as categorical values, which is not supposed
    hex_attributes = ["AppNonce", "DLSettings", "FPort", "MIC", "FCtrl", "FCnt", "FOpts", "MHDR", 
                      "CFList", "CFListType", "RxDelay"]

    df = DataPreProcessing.hex_to_decimal(df, hex_attributes)

    # add the label column with the value 0 (that will be later updated when adding intrusions on some testing samples)
    return df.withColumn("intrusion", lit(0))


"""
Applies pre-processing steps for all "rxpk" and "txpk" rows of the dataframe for
a specific device

"""
def prepare_df_for_device(spark_session, dataset_type, dev_addr, model_type, datasets_format):

    start_time = time.time()

    if datasets_format == "json":
        df_model = spark_session.read.json(
            f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset.{datasets_format}'
        ).filter(col("DevAddr") == dev_addr)

    else:
        df_model = spark_session.read.parquet(
            f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset.{datasets_format}'
        ).filter(col("DevAddr") == dev_addr)

    df_model = df_model.cache()     # Applies cache operations to speed up the processing

    # Calls function 'train_test_split' to verify if there are at least 'n_limit' samples on the dataset
    # to split for training and testing and get a effective model
    n_limit = 30

    # If there are not enough samples for the device, it's not possible to create a model since there is
    # no data to be used to train the model
    if df_model.count() < n_limit:
        print(f'There are no enough samples for the device {dev_addr} for {dataset_type.value["name"].upper()}. Must be at least {n_limit}. No model will be created.\n\n\n')
        return None, None

    # Remove columns of device dataset where all values are null
    non_null_columns = [
        c for c in df_model.columns
        if (
            # Check if NOT all values are null
            (df_model.agg(sum(when(col(c).isNotNull(), 1).otherwise(0))).first()[0] or 0) > 0
        )
    ]

    df_model = df_model.select(non_null_columns)

    # Remove columns from the string list that are not used for machine learning
    non_null_columns = list(set(non_null_columns) - set(["DevAddr", "intrusion"]))
    
    # replace NULL and empty values with the mean on numeric attributes with missing values, because these are values
    # that can assume any numeric value, so it's not a good approach to replace missing values with a static value
    # the mean is the best approach to preserve the distribution and variety of the data
    imputer = Imputer(inputCols=non_null_columns, outputCols=non_null_columns, strategy="mean")

    df_model = imputer.fit(df_model).transform(df_model)

    end_time = time.time()

    # Print the total time of pre-processing
    print(f'Total time of pre-processing in device {dev_addr} and {dataset_type.value["name"].upper()}: {format_time(end_time - start_time)} \n')

    # Applies division of samples into training and testing after processing dataframe 'df_model'
    df_model_train, df_model_test = train_test_split(df_model, seed=42)

    # NOTE: uncomment this line to print the number of training samples for the device
    print(f'Number of {dataset_type.value["name"].upper()} training samples for device {dev_addr}: {df_model_train.count()}')

    # ensure that, regardless of the size of the test dataset, we always insert between 1 and 12 intrusions,
    # and the number of intrusions is higher in larger datasets
    num_intrusions = min(round(0.2 * df_model_test.count()), 12)

    df_model_test = modify_device_dataset(df_train=df_model_train,
                                          df_test=df_model_test,
                                          params=["SF", "BW", "PHYPayloadLen"], 
                                          target_values=[SF_LIST, BW_LIST, DATA_LEN_LIST_ABNORMAL],
                                          num_intrusions=num_intrusions)

    # NOTE: uncomment this line to print the number of testing samples for the device
    print(f'Number of {dataset_type.value["name"].upper()} testing samples for device {dev_addr}: {df_model_test.count()}')

    return DataPreProcessing.features_assembler(df_model_train, df_model_test, model_type)

