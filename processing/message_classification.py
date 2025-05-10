
import time, mlflow, shutil, os
from common.dataset_type import DatasetType
from prepareData.prepareData import prepare_df_for_device
from common.auxiliary_functions import format_time
from common.constants import SPARK_PRE_PROCESSING_NUM_PARTITIONS
from mlflow.tracking import MlflowClient
from models.autoencoder import Autoencoder
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
import mlflow.pyspark.ml as mlflow_pyspark_ml
from mlflow.models import infer_signature
from models.kNN import KNNClassifier
from pyspark.sql.streaming import DataStreamReader


class MessageClassification:

    def __init__(self, spark_session):
        self.__spark_session = spark_session
        self.__mlflowclient = MlflowClient()


    """
    This function returns the MLFlow model based on the associated DevAddr, received in the
    parameter

    It returns a tuple with the model itself as a MLFlow artifact, and the id of the corresponding
    run in MLFlow
    
    """
    def __get_model_by_devaddr_and_dataset_type(self, dev_addr, dataset_type):
        
        # Search the model according to DevAddr
        runs = self.__mlflowclient.search_runs(
            experiment_ids=["0"],
            filter_string=f"tags.DevAddr = '{dev_addr}' and tags.MessageType = '{dataset_type}'",
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
    Auxiliary function that stores on MLFlow the model based on DevAddr, replacing the old model if it exists
    This model is used on training and testing, whether on only creating models based on past data, or for classifying
    new messages in real time, or for the re-training process
    
    """
    def __store_model(self, dev_addr, df_model_train, model, accuracy, dataset_type):

        # Verify if a model associated to the device already exists. If so, return it;
        # otherwise, return None
        mlflow_retrieved_model, old_run_id = self.__get_model_by_devaddr_and_dataset_type(
            dev_addr, dataset_type.value["name"]
        )

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

            # If the path exists (which happens in normal cases), delete it
            if os.path.exists(run_path):
                shutil.rmtree(run_path)
                print(f"Artefact directory deleted: {run_path}")

            else:
                print(f"Artefact directory not found: {run_path}")

        # Create model based on DevAddr and store it as an artifact using MLFlow
        with mlflow.start_run(run_name=f'Model_Device_{dev_addr}_{dataset_type.value["name"]}'):
            mlflow.set_tag("DevAddr", dev_addr)
            mlflow.set_tag("MessageType", dataset_type)
            mlflow_pyspark_ml.autolog()
            
            # for every algorithms except kNN, store the model as artifact
            mlflow.spark.log_model(model, "model")

            # for kNN, store the dataframe as a parquet file
            #dataset_model_path = f'./generatedDatasets/model_{dev_addr}_{dataset_type.value["name"]}.parquet'
            #df_model_train.write.mode("overwrite").parquet(dataset_model_path)
            #mlflow.log_artifact(dataset_model_path)
            
            if accuracy is not None:
                mlflow.log_metric("accuracy", accuracy)


    """
    This function trains or re-trains a ML model based on a given DevAddr, and stores it on MLFlow
    It uses, as input, all samples of the pandas dataframe 'df' whose DevAddr is equal to 'dev_addr'
    This function is also used when the IDS receives a new message in real time and the model for the DevAddr
    of the message doesn't exist yet

    """
    def __create_model(self, df_model_train, df_model_test, dev_addr, dataset_type):

        start_time = time.time()

        # Apply autoencoder to build a label based on the reconstruction error

        ae = Autoencoder(self.__spark_session, df_model_train, df_model_test, dev_addr)

        ae.train()

        df_model_train, df_model_test = ae.label_data_by_reconstruction_error()

        """# KNN
        knn = KNNClassifier(k=20, train_df=df_model_train,
                            test_df=df_model_test, featuresCol="features", 
                            labelCol="intrusion")

        results = knn.test()
        accuracy = results["accuracy"]"""
        

        # RANDOM FOREST
        
        # Apply Random Forest to detect intrusions based on the created label on Autoencoder
        rf = RandomForestClassifier(numTrees=30, featuresCol="features", labelCol="intrusion")
        model = rf.fit(df_model_train)

        if df_model_test is not None:
            results = model.evaluate(df_model_test)
            accuracy = results.accuracy
        else:
            accuracy = None

        
        """# LOGISTIC REGRESSION
        
        lr = LogisticRegression(featuresCol="features", labelCol="intrusion", 
                                regParam=0.1, elasticNetParam=1.0,
                                family="multinomial", maxIter=50)

        model = lr.fit(df_model_train)

        if df_model_test is not None:
            results = model.evaluate(df_model_test)
            accuracy = results.accuracy"""
        
        
        """  # KMEANS

        # Apply clustering (KMeans or, as alternative, DBSCAN) to divide samples into clusters according to the density
        k_means = KMeans(k=3, seed=522, maxIter=100)

        # TODO: think if covering this with a try/except block wouldn't be better
        model = k_means.fit(df_train)

        predictions = model.transform(df_test)
        
        # Evaluate the model
        evaluator = ClusteringEvaluator()
        
        accuracy = evaluator.evaluate(predictions)

        """

        
        if results is not None:
            print(f"accuracy for model of device {dev_addr}: {round((accuracy * 100), 2)}%")

        self.__store_model(dev_addr, df_model_train, model, accuracy, dataset_type)

        end_time = time.time()

        print(f'Model for end-device with DevAddr {dev_addr} and {dataset_type.value["name"]} saved successfully and created in {format_time(end_time - start_time)}')


    """
    Function to create models for some of all devices

    It receives the spark session (spark_session) that handles the dataset processing and
    the corresponding dataset type (dataset_type) defined by DatasetType Enum

    It stores the models as artifacts using MLFlow, as well as their associated informations 
    such as metric evaluations and the associated DevAddr 

        dev_addr_list - an optional parameter to specify, as a list of strings, the DevAddr of the devices
                        from which the user pretends to create models; if not specified, models from all 
                        devices will be created

    """
    def create_ml_models(self, dev_addr_list):

        ### Begin processing
        start_time = time.time()

        # create each model in sequence
        for dev_addr in dev_addr_list:
            for dt in DatasetType:

                df_train = self.__spark_session.read.json(
                    f'./generatedDatasets/{dt.value["name"]}/lorawan_dataset_train.json'
                )

                df_test = self.__spark_session.read.json(
                    f'./generatedDatasets/{dt.value["name"]}/lorawan_dataset_test.json'
                )

                df_model_train, df_model_test = prepare_df_for_device(df_train, df_test, dev_addr)  # Pre-Processing

                self.__create_model(df_model_train, df_model_test, dev_addr, dt)         # Processing
        
        end_time = time.time()

        # Print the total time; the time is in seconds, minutes or hours
        print("Total time of pre-processing + processing:", format_time(end_time - start_time), "\n\n")


    """
    Auxiliary function that classifies messages in real time, using the model that corresponds
    to the message's dev_addr

    TODO: add more parameters to the function if necessary
    
    """
    def classify_new_incoming_messages(self):

        # TODO: review step 0
        # Read stream from a socket (e.g., port 9999)
        """socket_stream_df = self.__spark_session.readStream \
                                .format("socket") \
                                .option("host", "localhost") \
                                .option("port", 9999) \
                                .load()"""

        # TODO
            # 0 - uses spark session (self.__spark_session) to open a socket where new messages are listened
            # 1 - reads the message (see how to do it later)
            # 2 - converts the message to a dataframe row
            # 3 - apply pre-processing on the received message calling the function "prepare_dataframe(df)"
            # 4 - classify the message using the corresponding model retrieved from MLFlow, based on DevAddr
            #       4a - call self.__get_model_by_dev_addr with the given parameters to check if the model exists
            #       4b - if the model does not exist, create it (self.__create_model) and store it on MLFlow; there will be no replaced model
            #                since the created model to be stored on MLFlow is the first one
            #       4c - use "predict" to classify the message using the corresponding model
            # 5 - aggregate the received and classified messages in a dataframe 'df_new_msgs' using 'union'
            # 6 - after receiving and classifying X messages (100, 200, etc), re-train the model calling function fit(df_new_msgs)
            #       6a - retrieve old model from MLFlow using self.__get_model_by_dev_addr
            #       6b - then just call "fit" and "transform" using the retrieved model from MLFlow, then replace the old model with the new model,
            #               calling self.__store_model; this allows the new model to learn new patterns (new intrusions) from the new data, the new
            #               LoRaWAN messages
            # 7 - it eventually waits to Ctrl + C or something, to close the socket
        
        pass

