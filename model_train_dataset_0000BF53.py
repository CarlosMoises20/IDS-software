
from common.spark_functions import create_spark_session
from pyspark.sql.functions import length, col, regexp_extract, lit
from pyspark.sql.types import IntegerType
from common.spark_functions import get_all_attributes_names, train_test_split
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import Window
import random
from models.kNN import KNNClassifier
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from models.autoencoder import Autoencoder
from models.isolation_forest import IsolationForest


# device 0000BF53

def modify_dataset(df, output_file_path, params, target_values):
    
    # Amostra aleatória de 25% das linhas
    df_to_modify = df.sample(fraction=0.25, seed=42)

    # Modify selected rows
    for param in params:
        random_value = random.choice(target_values)
        df_to_modify = df_to_modify.withColumn(param, lit(random_value))

    df_to_modify = df_to_modify.withColumn("intrusion", lit(1))

    # Get unmodified rows by filtering out selected row_numbers
    df_unmodified = df.join(df_to_modify, on=df.columns, how="left_anti")

    # Combine both parts
    df_final = df_unmodified.unionByName(df_to_modify)

    # Save the result in JSON file
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

    df_model_train, df_model_test = train_test_split(df_model)

    output_path = "./generatedDatasets/dataset_3_modified.json"

    df_model_test = modify_dataset(df=df_model_test,
                                    output_file_path=output_path,
                                    params=["SF", "BW"],
                                    target_values=[15, 1])


    if_class = IsolationForest(df_train=df_model_train, 
                               df_test=df_model_test, 
                               featuresCol="features",
                               labelCol="intrusion")

    if_class.train()

    predictions = if_class.test()

    accuracy, cm = if_class.evaluate()

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Apply autoencoder to build a label based on the reconstruction error
    """ae = Autoencoder(spark_session, df_model_train, df_model_test, "0000BF53", None)

    ae.train()

    df_model_test, accuracy, confusion_matrix = ae.test()

    print(accuracy)
    print(confusion_matrix)"""

    
    ### KNN 
    """knn = KNNClassifier(k=30, train_df=df_model_train,
                        test_df=df_model_test, featuresCol="features", 
                        labelCol="intrusion")
                        
    accuracy, matrix, labels, report = knn.test()

    if accuracy is not None:
        print(f'accuracy for model of device "0000BF53": {round((accuracy * 100), 2)}%')

    if matrix is not None:
        print("Confusion matrix:\n", matrix) 

    if labels is not None:
        print("Labels:", labels) 

    if report is not None:
        print("Report:\n", report)"""

    ### RANDOM FOREST
    #algorithm = RandomForestClassifier(numTrees=30, featuresCol="features", labelCol="intrusion")

    ### LOGISTIC REGRESSION
    """algorithm = LogisticRegression(featuresCol="features", labelCol="intrusion",
                            family="multinomial", maxIter=50)"""

    #model = algorithm.fit(df_model_train)

    """results = model.evaluate(df_model_test)
    accuracy = results.accuracy
    labels = results.labels
    precisionByLabel = results.precisionByLabel
    recallByLabel = results.recallByLabel
    falsePositiveRateByLabel = results.falsePositiveRateByLabel"""

    """if accuracy is not None:
        print(f'accuracy for model of device "0000BF53": {round((accuracy * 100), 2)}%')
    
    if labels is not None:
        print("Labels:", labels) 

    if precisionByLabel is not None:
        print("Precision By Label:", precisionByLabel)

    if recallByLabel is not None:
        print("Recall by label:", recallByLabel)

    if falsePositiveRateByLabel is not None:
        print("False Positive Rate By Label:", falsePositiveRateByLabel)"""