
import time, mlflow, shutil, os, json
import mlflow.sklearn
from common.dataset_type import DatasetType
from common.constants import SF_LIST, BW_LIST, DATA_LEN_LIST_ABNORMAL
from prepareData.prepareData import prepare_df_for_device
from common.auxiliary_functions import format_time
from mlflow.tracking import MlflowClient
from pyspark.sql.functions import col
from models.autoencoder import Autoencoder
from models.kNN import KNNAnomalyDetector
from models.one_class_svm import OneClassSVM
from pyspark.sql.streaming import DataStreamReader
from models.isolation_forest import IsolationForest
from common.spark_functions import modify_device_dataset

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
        
        # Search the model according to DevAddr and MessageType
        runs = self.__mlflowclient.search_runs(
            experiment_ids=["0"],
            filter_string=f"tags.DevAddr = '{dev_addr}' and tags.MessageType = '{dataset_type}'",
            order_by=["metrics.accuracy DESC"],  # or another criteria
            max_results=1
        )
        
        if not runs:
            return None, None

        run_id = runs[0].info.run_id
        
        try:
            model = mlflow.spark.load_model(f"./mlruns/0/{run_id}/artifacts/model")
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
        _, old_run_id = self.__get_model_by_devaddr_and_dataset_type(
            dev_addr, dataset_type.value["name"].lower()
        )

        # If a model associated to the device already exists, delete it to replace it with
        # the new model, so that the system is always with the newest model in order to 
        # be constantly learning new network traffic patterns
        if old_run_id is not None:
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
        with mlflow.start_run(run_name=f'Model_Device_{dev_addr}_{dataset_type.value["name"].lower()}'):
            mlflow.set_tag("DevAddr", dev_addr)
            mlflow.set_tag("MessageType", dataset_type.value["name"].lower())
            
            # for every algorithms except kNN, store the model as artifact
            #if model is not None:

                ## FOR ISOLATION FOREST
                #mlflow.sklearn.log_model(model, "model")

                ## FOR THE OTHER MODELS
                #mlflow.spark.log_model(model, "model") 

                #print(model)

                #mlflow.pytorch.log_model(model, "model") 

            # store the training dataframe as a parquet file
            dataset_model_path = f'./df_tmp/model_{dev_addr}_{dataset_type.value["name"]}.parquet'
            df_model_train.write.mode("overwrite").parquet(dataset_model_path)
            mlflow.log_artifact(dataset_model_path, artifact_path="training_dataset")
            
            if accuracy is not None:
                mlflow.log_metric("accuracy", accuracy)


    """
    This function trains or re-trains a ML model based on a given DevAddr, and stores it on MLFlow
    It uses, as input, all samples of the pandas dataframe 'df' whose DevAddr is equal to 'dev_addr'
    This function is also used when the IDS receives a new message in real time and the model for the DevAddr
    of the message doesn't exist yet

    """
    def __create_model(self, df_model_train, df_model_test, dev_addr, dataset_type, intrusion_rate):

        start_time = time.time()

        """### ISOLATION FOREST
        
        if_class = IsolationForest(spark_session=self.__spark_session,
                                   df_train=df_model_train, 
                                   df_test=df_model_test, 
                                   featuresCol="features",
                                   predictionCol="prediction",
                                   labelCol="intrusion",
                                   contamination=intrusion_rate)

        model = if_class.train()

        df_predictions = if_class.test(model)

        accuracy, matrix = if_class.evaluate(df_predictions)"""
        
        """### AUTOENCODER (not good results on detecting intrusions)
        
        ae = Autoencoder(df_train=df_model_train, df_test=df_model_test)

        model = ae.train()

        accuracy, matrix, report = ae.test()"""

        """### kNN (very innefficient with large datasets)
        
        model, accuracy, matrix, labels, report = None, None, None, None, None

        ### KNN to detect anomalies (not as tradicional binary classifier)
        knn = KNNAnomalyDetector(k=5, df_train=df_model_train,
                                df_test=df_model_test, 
                                featuresCol="features", 
                                labelCol="intrusion",
                                predictionCol="prediction",
                                threshold_percentile=95)

        model = knn.train()

        accuracy, matrix, report = knn.test(model)"""


        ### One-Class SVM

        ocsvm = OneClassSVM(spark_session=self.__spark_session,
                            df_train=df_model_train,
                            df_test=df_model_test,
                            featuresCol="features",
                            predictionCol="prediction",
                            labelCol="intrusion")
        
        model = ocsvm.train()

        df_preds = ocsvm.test(model)

        accuracy, evaluation = ocsvm.evaluate(df_preds)

        if evaluation is not None:
            print("Evaluation Report:\n")
            for key, value in evaluation.items():
                print(f"{key}: {value}")
        
        """if accuracy is not None:
            print(f'accuracy for model of device {dev_addr} for {dataset_type.value["name"].upper()}: {round((accuracy * 100), 2)}%')
        
        if matrix is not None:
            print("Confusion matrix:\n", matrix) 
        
        if report is not None:
            print("Report:\n", json.dumps(report, indent=4))
            #print("Report:\n", report)"""

        self.__store_model(dev_addr, df_model_train, model, accuracy, dataset_type)

        end_time = time.time()

        print(f'Model for end-device with DevAddr {dev_addr} and {dataset_type.value["name"].upper()} saved successfully and created in {format_time(end_time - start_time)}\n\n\n')


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
    def create_ml_models(self, dev_addr_list, datasets_format):

        ### Begin processing
        start_time = time.time()

        # create each model in sequence
        for dev_addr in dev_addr_list:

            for dataset_type in DatasetType:

                # Pre-Processing phase
                df_model_train, df_model_test = prepare_df_for_device(self.__spark_session, dataset_type, dev_addr)  

                # If there are samples for the device, the model will be created
                if (df_model_train, df_model_test) != (None, None):

                    dataset_path = f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_{dev_addr}_test.{datasets_format}'

                    # TODO try put all possible range and then on 'modify_device_dataset', only apply
                    # the values that are inside the array and are not inside the device dataset 

                    num_intrusions = 10

                    intrusion_rate = num_intrusions / df_model_test.count()

                    df_model_test = modify_device_dataset(df=df_model_test,
                                                            output_file_path=dataset_path,
                                                            params=["SF", "BW", "dataLen"], 
                                                            target_values=[SF_LIST, BW_LIST, DATA_LEN_LIST_ABNORMAL],
                                                            datasets_format=datasets_format,
                                                            num_intrusions=num_intrusions)
                    
                    # Processing phase
                    self.__create_model(df_model_train, df_model_test, dev_addr, dataset_type, intrusion_rate)         
        
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
            # 4 - classify the message using the corresponding model retrieved from MLFlow, based on DevAddr and message type
            #       4a - call self.__get_model_by_dev_addr with the given parameters to check if the model exists
            #       4b - if the model does not exist, create it (self.__create_model) and store it on MLFlow; there will be no replaced model
            #                since the created model to be stored on MLFlow is the first one
            #       4c - use "predict" to classify the message using the corresponding model
            # 5 - aggregate the received and classified messages in a dataframe 'df_new_msgs' using 'union'
            # 6 - after receiving and classifying X messages (100, 200, etc), re-train the model
            #       6a - retrieve old model from MLFlow using self.__get_model_by_dev_addr
            #       6b - download the artifact corresponding to the dataset used to train the old model
            #           """mlflow.artifacts.download_artifacts(run_id=<RUN_ID>, artifact_path="training_dataset")"""
            #       6c - convert the dataset to a dataframe and bind it to the new messages
            #       6d - use the binded dataframe to create a new model that will replace the old model, calling "fit" on an already processed and labelled dataset
            #       6e - replace the old model calling self.__store_model; this allows the new stored model to learn new patterns (new intrusions) from the new data, the new
            #               LoRaWAN messages; 
            #       6f - also store the new dataset used for training as an artifact, replacing the old dataset
            # 7 - it eventually waits to Ctrl + C or something, to close the socket
        
        pass

