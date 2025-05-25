
from common.spark_functions import create_spark_session
from pyspark.sql.functions import length, col, regexp_extract, lit, monotonically_increasing_id, row_number
from pyspark.sql.types import IntegerType
from common.spark_functions import get_all_attributes_names, sample_random_split
from launch_attacks import modify_parameters
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import Window
import random
from models.kNN import KNNClassifier
from models.autoencoder import Autoencoder


# device 0000BF53

def modify_dataset(df, output_file_path, num_samples, params, target_values):
    
    # Add a global row number for arbitrary selection
    window_spec = Window.orderBy(monotonically_increasing_id())
    df_with_rownum = df.withColumn("row_number", row_number().over(window_spec))

    # Select N arbitrary rows to modify
    df_to_modify = df_with_rownum.filter(col("row_number") <= num_samples)

    # Modify selected rows
    for param in params:
        random_value = random.choice(target_values)
        df_to_modify = df_to_modify.withColumn(param, lit(random_value))

    df_to_modify = df_to_modify.withColumn("intrusion", lit(1))

    # Get unmodified rows by filtering out selected row_numbers
    df_unmodified = df_with_rownum.join(
        df_to_modify.select("row_number"), on="row_number", how="left_anti"
    )

    # Drop helper column and combine both parts
    df_to_modify = df_to_modify.drop("row_number")
    df_unmodified = df_unmodified.drop("row_number")
    df_final = df_unmodified.unionByName(df_to_modify)

    # Write to file
    df_final.coalesce(1).write.mode("overwrite").json(output_file_path)

    return df_final


if __name__ == '__main__':

    spark_session = create_spark_session()

    file_path = "./datasets/dataset_3_original.log"

    df_model = spark_session.read.json(file_path)

    df_model = df_model.cache()

    df_model = df_model.withColumn("payloadLen", length(col("payload"))) \
                        .withColumn("intrusion", col("flag"))

    df_model = df_model.drop("devaddr", "time", "payload", "flag")

    ### Extract values of LoRa parameters SF and BW from "datr" attribute
    pattern = r"SF(\d+)BW(\d+)"     # regex pattern to extract "SF" and "BW" 

    df_model = df_model.withColumn("SF", regexp_extract(col("datr"), pattern, 1).cast(IntegerType())) \
                        .withColumn("BW", regexp_extract(col("datr"), pattern, 2).cast(IntegerType()))

    # Remove "datr" after splitting it by "SF" and "BW"
    df_model = df_model.drop("datr")

    # Normalize all attributes except DevAddr that will not be used for model training, only to identify the model
    column_names = list(set(get_all_attributes_names(df_model.schema)) - set(["devaddr", "intrusion"]))

    assembler = VectorAssembler(inputCols=column_names, outputCol="feat")

    df_model = assembler.transform(df_model)

    scaler = MinMaxScaler(inputCol="feat", outputCol="features")

    df_model = scaler.fit(df_model).transform(df_model)

    df_model = df_model.drop("feat")

    df_model_train, df_model_test = sample_random_split(df_model)

    output_path = "./generatedDatasets/dataset_3_modified.json"

    df_model_test = modify_dataset(df=df_model_test,
                                    output_file_path=output_path,
                                    num_samples=30,
                                    params=["SF"],
                                    target_values=[2])

    # TODO fix
    """modify_parameters(spark_session=spark_session,
                      file_path=output_path,
                      avg_num_samples_per_device=10,
                      params=["payloadLen"],
                      target_values=[400],
                      dataset_format="json")"""


    # Apply autoencoder to build a label based on the reconstruction error
    #ae = Autoencoder(spark_session, df_model_train, df_model_test, "0000BF53", None)

    #ae.train()

    #df_model_test = ae.label_data_by_reconstruction_error()

    #df_model_train = df_model_train.filter(col("intrusion") == 0)

    
    ### KNN 
    knn = KNNClassifier(k=15, train_df=df_model_train,
                        test_df=df_model_test, featuresCol="features", 
                        labelCol="intrusion")
                        
    accuracy = knn.test()

    print(accuracy)