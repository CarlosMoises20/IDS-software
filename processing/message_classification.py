
import time
from prepareData.prepareData import prepare_data
from common.auxiliary_functions import format_time
from common.constants import SPARK_PROCESSING_NUM_PARTITIONS
from mlflow.tracking import MlflowClient
from models.autoencoder import Autoencoder
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
import mlflow.pyspark.ml as mlflow_pyspark_ml
from mlflow.models import infer_signature
import mlflow, shutil, os
from pyspark.sql.streaming import DataStreamReader


class MessageClassification:

    def __init__(self, spark_session):
        self.__spark_session = spark_session
        self.__mlflowclient = MlflowClient()


    """
    This function returns the MLFlow model based on the associated DevAddr, received in the
    parameter
    
    """
    def __get_model_by_devaddr(self, dev_addr):
        
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
    considering the total number of examples in the dataframe corresponding to the device, if
    the total number of samples is larger than 1

    """
    def __sample_random_split(self, df_model, seed):

        # Count the total number of samples to be used by the model
        total_count = df_model.count()

        # If there is only one sample for the device, use that sample for training, 
        # and don't apply testing for that model
        if total_count == 1:
            df_model_train, df_model_test = df_model, None

        # If there are between 2 and 9 samples, split the samples for training and testing by 50-50
        elif total_count < 10:
            df_model_train, df_model_test = df_model.randomSplit([0.5, 0.5], seed=seed)

        # If there are between 10 and 20 samples, split the samples for training and testing by 70-30
        elif total_count < 20:
            df_model_train, df_model_test = df_model.randomSplit([0.7, 0.3], seed=seed)

        # If there are 20 or more samples, split the samples for training and testing by 85-15
        else:
            df_model_train, df_model_test = df_model.randomSplit([0.85, 0.15], seed=seed)

        return df_model_train, df_model_test


    """
    Auxiliary function that stores on MLFlow the model based on DevAddr, replacing the old model if it exists
    This model is used on training and testing, whether on only creating models based on past data, or for classifying
    new messages in real time, or for the re-training process
    
    """
    def __store_model(self, dev_addr, df, df_model_test, model, accuracy):

        # Verify if a model associated to the device already exists. If so, return it;
        # otherwise, return None
        mlflow_retrieved_model, old_run_id = self.__get_model_by_devaddr(dev_addr)

        signature = infer_signature(df_model_test, df)

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
        with mlflow.start_run(run_name=f"Model_Device_{dev_addr}"):
            mlflow.set_tag("DevAddr", dev_addr)
            mlflow.log_metric("accuracy", accuracy)
            mlflow_pyspark_ml.autolog()
            mlflow.spark.log_model(model, "model", signature=signature)


    """
    This function trains or re-trains a ML model based on a given DevAddr, and stores it on MLFlow
    It uses, as input, all samples of the pandas dataframe 'df' whose DevAddr is equal to 'dev_addr'
    This function is also used when the IDS receives a new message in real time and the model for the DevAddr
    of the message doesn't exist yet

    """
    def create_model(self, df_model, dev_addr):

        start_time = time.time()

        ae = Autoencoder(self.__spark_session, df_model, dev_addr)

        ae.train()

        df = ae.label_data_by_reconstruction_error()

        # randomly divide dataset into training and test, according to the total number of examples 
        # and set a seed in order to ensure reproducibility, which is important to 
        # ensure that the model is always trained and tested on the same examples each time the
        # model is run. This is important to compare the model's performance in different situations
        df_model_train, df_model_test = self.__sample_random_split(df_model=df, seed=522)

        # Apply Random Forest to detect intrusions based on the created label on Autoencoder
        rf = RandomForestClassifier(numTrees=30, featuresCol="features", labelCol="intrusion")
        rf_model = rf.fit(df_model_train)

        if df_model_test is not None:
            results = rf_model.evaluate(df_model_test)
            accuracy = results.accuracy

        
        """ LOGISTIC REGRESSION
        
        lr = LogisticRegression(featuresCol="features", labelCol="intrusion", 
                                regParam=0.1, elasticNetParam=1.0,
                                family="multinomial", maxIter=50)
        

        lr_model = lr.fit(df_train)

        results = lr_model.evaluate(df_test)
        
        """
        
        
        """  KMEANS

        # Apply clustering (KMeans or, as alternative, DBSCAN) to divide samples into clusters according to the density
        k_means = KMeans(k=3, seed=522, maxIter=100)

        # TODO: think if covering this with a try/except block wouldn't be better
        model = k_means.fit(df_train)

        predictions = model.transform(df_test)
        
        # Evaluate the model
        evaluator = ClusteringEvaluator()
        
        accuracy = evaluator.evaluate(predictions)

        accuracy_list.append(accuracy)

        """

        
        if results is not None:
            print(f"accuracy for model of device {dev_addr}: {round((accuracy * 100), 2)}%")

        # Store the model on MLFlow and replace the old model if it exists
        self.__store_model(dev_addr, df, df_model_test, rf_model, accuracy)

        end_time = time.time()

        print(f"Model for end-device with DevAddr {dev_addr} saved successfully and created in {format_time(end_time - start_time)}")


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
        df = prepare_data(self.__spark_session)

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


        # create each model in sequence
        for dev_addr in dev_addr_list:
            df_model = df.filter(df.DevAddr == dev_addr)
            self.create_model(df_model, dev_addr)
        
    
        end_time = time.time()

        # Print the total time of pre-processing; the time is in seconds, minutes or hours
        print("Total time of processing:", format_time(end_time - start_time), "\n\n")



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
            #       4a - if the model does not exist, create it 
            #                 (call self.__store_model with the given parameters)
            #       4b - use "predict" to classify the message using the corresponding model
            # 5 - aggregate the received and classified messages in a dataframe 'df_new_msgs' using 'union'
            # 6 - after receiving and classifying X messages (100, 200, etc), re-train the model calling function fit(df_new_msgs)
            # 7 - it eventually waits to Ctrl + C or something, to close the socket
        
        pass

