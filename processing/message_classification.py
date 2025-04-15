
from common.auxiliary_functions import format_time
import mlflow, time, shutil, os
import mlflow.pytorch as mlflow_pytorch
import mlflow.pyspark.ml as mlflow_pyspark
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from prepareData.prepareData import prepare_past_dataset
from models.autoencoder import Autoencoder
from common.constants import SPARK_NUM_PARTITIONS
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from concurrent.futures import ProcessPoolExecutor


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
    def sample_random_split(self, df_model):

        total_count = df_model.count()

        if total_count < 10:
            df_model_train, df_model_test = df_model.randomSplit([0.5, 0.5], seed=522)

        elif total_count < 20:
            df_model_train, df_model_test = df_model.randomSplit([0.7, 0.3], seed=522)

        else:
            df_model_train, df_model_test = df_model.randomSplit([0.85, 0.15], seed=522)

        return df_model_train, df_model_test


    """
    This function creates a ML model based on a given DevAddr, and stores it on MLFlow
    It uses, as input, all samples of the dataframe 'df' whose DevAddr is equal to 'dev_addr'

    """
    def create_model(self, df, dev_addr, accuracy_list=None):
        
        # Filter dataset considering the selected DevAddr
        # Remove DevAddr to make processing more efficient, since we don't need it anymore 
        df_model = df.filter(df.DevAddr == dev_addr).drop("DevAddr")

        ae = Autoencoder(self.__spark_session, df_model, dev_addr)

        ae.train()

        df_model = ae.label_data_by_reconstruction_error(threshold=0.0375)


        # randomly divide dataset into training and test, according to the total number of examples 
        # and set a seed in order to ensure reproducibility, which is important to 
        # ensure that the model is always trained and tested on the same examples each time the
        # model is run. This is important to compare the model's performance in different situations
        df_model_train, df_model_test = self.sample_random_split(df_model)


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
        results.predictions.show(15, truncate=False, vertical=True)
        
        
        


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

        """  SAVE MODEL ON MLFLOW 
        # Verify if a model associated to the device already exists. If so, return it;
        # otherwise, return None
        mlflow_retrieved_model, old_run_id = self.get_model_by_devaddr(dev_addr)

        signature = infer_signature(df_model_test, df_model)

        # If a model associated to the device already exists, delete it to replace it with
        # the new model, so that the system is always with the newest model in order to 
        # be constantly learning new network traffic patterns
        if mlflow_retrieved_model is not None:
            self.__mlflowclient.delete_run(old_run_id)
            print(f"Old model from device {dev_addr} deleted.")

            # Get experiment ID of run
            run_info = mlflow.get_run(old_run_id).info
            experiment_id = run_info.experiment_id

            # Artefact local path
            run_path = os.path.join("mlruns", experiment_id, old_run_id)

            # If the path exists, delete it
            if os.path.exists(run_path):
                shutil.rmtree(run_path)
                print(f"Artefact directory deleted: {run_path}")

            else:
                print(f"Artefact directory not found: {run_path}")
  
        # Create model based on DevAddr and store it as an artifact using MLFlow
        with mlflow.start_run(run_name=f"Model_Device_{dev_addr}"):
            mlflow.set_tag("DevAddr", dev_addr)
            mlflow.log_metric("accuracy", accuracy)
            mlflow_pyspark.autolog()
            mlflow.spark.log_model(model, "model", signature=signature)


        print(f"Model for end-device with DevAddr {dev_addr} saved successfully")

        """
        


    """
    Function to execute the IDS

    It receives the spark session (spark_session) that handles the dataset processing and
    the corresponding dataset type (dataset_type) defined by DatasetType Enum

    It stores the models as artifacts using MLFlow, as well as their associated informations 
    such as metric evaluations and the associated DevAddr 

    """
    def create_ml_models(self):

        # pre-processing: prepare past dataset
        df = prepare_past_dataset(self.__spark_session)

        # Splits the dataframe into "SPARK_NUM_PARTITIONS" partitions during pre-processing
        df = df.coalesce(numPartitions=int(SPARK_NUM_PARTITIONS))

        ### Begin processing
        start_time = time.time()

        # Cast DevAddr column to integer and get distinct values
        dev_addr_list = df.select("DevAddr").filter(df["DevAddr"].isNotNull()).distinct()

        # Convert to a list of integers
        dev_addr_list = [row.DevAddr for row in dev_addr_list.collect()]

        # list of all models' accuracy to be used to return the mean accuracy of all models
        accuracy_list = []

        # create all models in parallel to accelerate the code execution, since these are all independent
        # and we are dealing with a lot of models (over 4000 models)
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.create_model(df, dev_addr, accuracy_list))
                       for dev_addr in dev_addr_list]
            
            # Waits until all threads get completed
            for future in futures:
                future.result()
    
        end_time = time.time()

        avg_accuracy = (sum(accuracy_list) / len(accuracy_list)) * 100

        print("Average accuracy_list:", avg_accuracy, "%")

        # Print the total time of pre-processing; the time is in seconds, minutes or hours
        print("Time of processing: ", format_time(end_time - start_time), "\n\n")

        
