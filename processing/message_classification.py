
import pandas as pd
import mlflow, time, shutil, os
import mlflow.pyspark.ml as mlflow_pyspark
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from prepareData.prepareData import prepare_past_dataset
from models.autoencoder import Autoencoder
from common.auxiliary_functions import format_time
from common.constants import SPARK_PROCESSING_NUM_PARTITIONS
from pyspark.sql.functions import col
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression


def get_model_by_devaddr(dev_addr, mlflowclient):
    
    # Search the model according to DevAddr
    runs = mlflowclient.search_runs(
        experiment_ids=["0"],
        filter_string=f"tags.DevAddr = '{dev_addr}'",
        order_by=["metrics.accuracy DESC"],  # ou outro critério
        max_results=1
    )
    
    if not runs:
        return None, None

    run_id = runs[0].info.run_id
    
    model_uri = f"runs:/{run_id}/model"
    
    try:
        model = mlflow.spark.load_model(model_uri)
    except:
        model = None

    return model, run_id

"""
This function ensures that there are always sufficient samples for both training and testing
considering the total number of examples in the dataframe corresponding to the device

"""
def sample_random_split(df_model):

    total_count = df_model.count()

    if total_count < 10:
        df_model_train, df_model_test = df_model.randomSplit([0.5, 0.5], seed=522)

    elif total_count < 20:
        df_model_train, df_model_test = df_model.randomSplit([0.7, 0.3], seed=522)

    else:
        df_model_train, df_model_test = df_model.randomSplit([0.85, 0.15], seed=522)

    return df_model_train, df_model_test


# Define the function that Spark will run in parallel per device's DevAddr
def model_train_udf(pdf):
    dev_addr = pdf["DevAddr"].iloc[0]
    print(dev_addr)
    create_model(pdf, dev_addr) 
    return pdf


"""
This function creates a ML model based on a given DevAddr, and stores it on MLFlow
It uses, as input, all samples of the pandas dataframe 'pdf' whose DevAddr is equal to 'dev_addr'

"""
def create_model(pdf, dev_addr):

    ae = Autoencoder(pdf, dev_addr)

    ae.train()

    #
    df_model = ae.label_data_by_reconstruction_error()

    # randomly divide dataset into training and test, according to the total number of examples 
    # and set a seed in order to ensure reproducibility, which is important to 
    # ensure that the model is always trained and tested on the same examples each time the
    # model is run. This is important to compare the model's performance in different situations
    df_model_train, df_model_test = sample_random_split(df_model)

    rf = RandomForestClassifier(numTrees=30, featuresCol="features", labelCol="intrusion")
    
    rf_model = rf.fit(df_model_train)

    results = rf_model.evaluate(df_model_test)
    
    """ LOGISTIC REGRESSION
    
    lr = LogisticRegression(featuresCol="features", labelCol="intrusion", 
                            regParam=0.1, elasticNetParam=1.0,
                            family="multinomial", maxIter=50)
    

    lr_model = lr.fit(df_model_train)

    results = lr_model.evaluate(df_model_test)
    
    """
    
    print(f"accuracy: {results.accuracy:.2f}")
    print(f"precision for each label: {results.precisionByLabel}")
    
    
    """  KMEANS

    # Apply clustering (KMeans or, as alternative, DBSCAN) to divide samples into clusters according to the density
    k_means = KMeans(k=3, seed=522, maxIter=100)

    # TODO: think if covering this with a try/except block wouldn't be better
    model = k_means.fit(df_model_train)

    predictions = model.transform(df_model_test)
    
    # Evaluate the model
    evaluator = ClusteringEvaluator()
    
    accuracy = evaluator.evaluate(predictions)

    accuracy_list.append(accuracy)

    """


    mlflowclient = MlflowClient()
    
    # Verify if a model associated to the device already exists. If so, return it;
    # otherwise, return None
    mlflow_retrieved_model, old_run_id = get_model_by_devaddr(dev_addr, mlflowclient)

    signature = infer_signature(df_model_test, df_model)

    # If a model associated to the device already exists, delete it to replace it with
    # the new model, so that the system is always with the newest model in order to 
    # be constantly learning new network traffic patterns
    if mlflow_retrieved_model is not None:
        mlflowclient.delete_run(old_run_id)
        print(f"Old model from device {dev_addr} deleted.")

        # Get experiment ID of run
        run_info = mlflow.get_run(old_run_id).info
        experiment_id = run_info.experiment_id

        # Artefact local path
        run_path = os.path.join("mlruns", experiment_id, old_run_id)

        # If the path exists (which happens in normal cases), delete it
        if os.path.exists(run_path):
            shutil.rmtree(run_path)
            print(f"Artefact directory deleted: {run_path}")

        else:
            print(f"Artefact directory not found: {run_path}")

    # Create model based on DevAddr and store it as an artifact using MLFlow
    with mlflow.start_run(run_name=f"Model_Device_{dev_addr}"):
        mlflow.set_tag("DevAddr", dev_addr)
        mlflow.log_metric("accuracy", results.accuracy)
        mlflow_pyspark.autolog()
        mlflow.spark.log_model(rf_model, "model", signature=signature)


    print(f"Model for end-device with DevAddr {dev_addr} saved successfully")


class MessageClassification:

    def __init__(self, spark_session):
        self.__spark_session = spark_session
        #self.__mlflowclient = MlflowClient()


    """
    Function to execute the IDS

    It receives the spark session (spark_session) that handles the dataset processing and
    the corresponding dataset type (dataset_type) defined by DatasetType Enum

    It stores the models as artifacts using MLFlow, as well as their associated informations 
    such as metric evaluations and the associated DevAddr 

        dev_addr_list - an optional parameter to specify, as a list of strings, the DevAddr of the devices
                        from which the user pretends to create models; if not specified, models from all 
                        devices will be created

    """
    def create_ml_models(self, dev_addr_list=None):

        # pre-processing: prepare past dataset
        df = prepare_past_dataset(self.__spark_session)

        # Splits the dataframe into "SPARK_PROCESSING_NUM_PARTITIONS" partitions during pre-processing
        df = df.coalesce(numPartitions=int(SPARK_PROCESSING_NUM_PARTITIONS))

        ### Begin processing
        start_time = time.time()

        # When dev_addr_list is not specified, models of all devices are created
        if dev_addr_list is None:

            # Cast DevAddr column to integer and get distinct values
            dev_addr_list = df.select("DevAddr").filter(df["DevAddr"].isNotNull()).distinct()

            # Convert to a list of integers
            dev_addr_list = [row.DevAddr for row in dev_addr_list.collect()]

        else:
            # When dev_addr_list is specified, remove, from the dataset, rows 
            # whose DevAddr does not belong to the list defined by the user
            df = df.filter(df.DevAddr.isin(dev_addr_list))

        # create all models in parallel to accelerate code execution
        result_df = df.groupBy("DevAddr").applyInPandas(model_train_udf, df.schema)

        # write the dataframe in a CSV file, excluding the column "features", because it's not necessary for visualization,
        # it's only used for ML algorithms' processing and its vector type is not supported by CSV
        # CSV has chosen since it's a simple-to-visualize and efficient format   
        result_df \
            .drop("features") \
            .write \
            .mode("overwrite") \
            .option("header", True) \
            .csv("./generatedDatasets/ids_final_results")
    
        end_time = time.time()

        # Print the total time of pre-processing; the time is in seconds, minutes or hours
        print("Total time of processing:", format_time(end_time - start_time), "\n\n")

        return result_df