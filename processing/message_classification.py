
from common.auxiliary_functions import format_time
from common.constants import CRATEDB_URI
import mlflow
import mlflow.pytorch as mlflow_pytorch
import mlflow.pyspark.ml as mlflow_pyspark
from mlflow.tracking import MlflowClient
from prepareData.prepareData import prepare_dataset
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark import SparkContext
import time


class MessageClassification:

    def __init__(self, spark_session):
        self.__spark_session = spark_session
        self.__mlflowclient = MlflowClient()


    def get_model_by_devaddr(self, dev_addr):
        
        # Search the model according to DevAddr
        runs = self.__mlflowclient.search_runs(
            experiment_ids=["0"],
            filter_string=f"tags.DevAddr = '{dev_addr}'",
            order_by=["metrics.accuracy DESC"],  # ou outro critério
            max_results=1
        )
        
        if not runs:
            return None

        run_id = runs[0].info.run_id
        
        model_uri = f"runs:/{run_id}/model"
        
        model = mlflow.spark.load_model(model_uri)
        
        return model

    # This function ensures that there are always sufficient samples for both training and testing
    # considering the total number of examples in the dataframe corresponding to the device
    def sample_random_split(self, df_model):

        total_count = df_model.count()

        if total_count < 10:
            df_model_train, df_model_test = df_model.randomSplit([0.5, 0.5], seed=522)

        elif total_count < 20:
            df_model_train, df_model_test = df_model.randomSplit([0.7, 0.3], seed=522)

        else:
            df_model_train, df_model_test = df_model.randomSplit([0.85, 0.15], seed=522)

        return df_model_train, df_model_test


    def __train_test_model(self, df, dev_addr, accuracy_list):
        
        # Filter dataset considering the selected DevAddr
        # Remove DevAddr to make processing more efficient, since we don't need it anymore 
        df_model = df.filter(df.DevAddr == dev_addr).drop("DevAddr")

        # randomly divide dataset into training and test, according to the total number of examples 
        # and set a seed in order to ensure reproducibility, which is important to 
        # ensure that the model is always trained and tested on the same examples each time the
        # model is run. This is important to compare the model's performance in different situations
        df_model_train, df_model_test = self.sample_random_split(df_model)

        # Apply clustering (KMeans or, as alternative, DBSCAN) to divide samples into clusters according to the density
        k_means = KMeans(k=2, seed=522, maxIter=100)

        model = k_means.fit(df_model_train)

        predictions = model.transform(df_model_test)

        predictions.show(truncate=False, vertical=True)
        
        # Evaluate the model
        evaluator = ClusteringEvaluator()
        
        accuracy = evaluator.evaluate(predictions)

        accuracy_list.append(accuracy)

        # Verify if a model associated to the device already exists. If so, return it;
        # otherwise, return None
        mlflow_retrieved_model = self.get_model_by_devaddr(dev_addr)

        # If a model associated to the device already exists, delete it to replace it with
        # the new model, so that the system is always with the newest model in order to 
        # be constantly learning new network traffic patterns
        if mlflow_retrieved_model is not None:
            old_run_id = mlflow_retrieved_model
            self.__mlflowclient.delete_run(old_run_id)
            print(f"Old model from device {dev_addr} deleted.")
  
        with mlflow.start_run(run_name=f"Model_{dev_addr})"):
            mlflow.set_tag("DevAddr", dev_addr)
            mlflow.log_metric("accuracy", accuracy)
            mlflow_pyspark.autolog()
            mlflow.spark.log_model(model, "model")

        print(f"Model for {dev_addr} saved successfully")
        


    """
    Function to execute the IDS

    It receives the spark session (spark_session) that handles the dataset processing and
    the corresponding dataset type (dataset_type) defined by DatasetType Enum

    It returns the processing results, namely the accuracy and the confusion matrix that show the
    model performance

    """
    def classify_messages(self):

        # pre-processing: prepare dataset
        df = prepare_dataset(self.__spark_session)

        # Divide dataframe processing into partitions to make it faster for ML processing
        df = df.repartition(200)

        ### Begin processing
        start_time = time.time()

        # Cast DevAddr column to integer and get distinct values
        dev_addr_list = df.select("DevAddr").filter(df["DevAddr"].isNotNull()).distinct()

        # Convert to a list of integers
        dev_addr_list = [row.DevAddr for row in dev_addr_list.collect()]

        # list of all models' accuracy to be used to return the mean accuracy of all models
        accuracy_list = []

        ### Initialize connection with CrateDB
        #db_connection = connect(CRATEDB_URI)

        #cursor = db_connection.cursor()

        # TODO: parallelize the creation of all models

        for dev_addr in dev_addr_list:
            self.__train_test_model(df, dev_addr, accuracy_list)

        # Close connection to CrateDB
        #cursor.close()
        #db_connection.close()

        end_time = time.time()

        avg_accuracy = (sum(accuracy_list) / len(accuracy_list)) * 100

        print("Average accuracy_list:", avg_accuracy, "%")

        # Print the total time of pre-processing; the time is in seconds, minutes or hours
        print("Time of processing: ", format_time(end_time - start_time), "\n\n")

        avg_accuracy = 30

        # TODO: change return for a general confusion matrix??
        return avg_accuracy

        
