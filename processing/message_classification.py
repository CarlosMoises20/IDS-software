
import time, mlflow, shutil, os, json, cloudpickle
import mlflow.sklearn
from common.dataset_type import DatasetType
from prepareData.prepareData import prepare_df_for_device
from common.auxiliary_functions import format_time
from pyspark.sql.functions import count
from mlflow.tracking import MlflowClient
from models.one_class_svm import OneClassSVM
from models.hbos import HBOS
from common.model_type import ModelType
from models.lof import LOF
from models.kNN import KNNAnomalyDetector
from pyspark.sql.streaming import DataStreamReader
from models.isolation_forest_custom import IsolationForest as CustomIF
from models.isolation_forest_sklearn import IsolationForest as SkLearnIF

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
    def __get_model_by_devaddr_and_dataset_type(self, dev_addr, dataset_type, model_type):
        
        # Search the model according to DevAddr and MessageType
        runs = self.__mlflowclient.search_runs(
            experiment_ids=["0"],
            filter_string=f"tags.DevAddr = '{dev_addr}' and tags.MessageType = '{dataset_type}'",
            max_results=1
        )
        
        if not runs:
            return None, None, None

        # for all
        run_id = runs[0].info.run_id
        model, model_id = None, None
        
        try:
            # for sklearn models
            if model_type == "sklearn":
                path = f"./mlruns/0/{run_id}/outputs"
                model_id = [name for name in os.listdir(path) 
                        if os.path.isdir(os.path.join(path, name)) and name.startswith("m-")]
                model_id = model_id[0] if model_id else None

                model = mlflow.sklearn.load_model(f"./mlruns/0/models/{model_id}/artifacts")
            
             # for spark models (kNN)
            elif model_type == "spark":
                model = mlflow.spark.load_model(f"./mlruns/0/{run_id}/artifacts/model")

            elif model_type == "pyod":
                model_uri = f'./mlruns/0/{run_id}/artifacts/model/model_{dev_addr}_{dataset_type}.pkl'
                with open(model_uri, "rb") as f:
                    model = cloudpickle.load(f)
            
        except:
            model, model_id = None, None

        print("model:", model)
        
        return model, run_id, model_id

    """
    Auxiliary function that stores on MLFlow the model based on DevAddr, replacing the old model if it exists
    This model is used on training and testing, whether on only creating models based on past data, or for classifying
    new messages in real time, or for the re-training process
    
    """
    def __store_model(self, dev_addr, dataset_train_path, dataset_test_path, model, model_type,
                      dataset_type, accuracy, matrix, recall_anomalies, report):

        # Verify if a model associated to the device already exists. If so, return it;
        # otherwise, return None
        _, old_run_id, old_model_id = self.__get_model_by_devaddr_and_dataset_type(
            dev_addr=dev_addr, 
            dataset_type=dataset_type.value["name"].lower(), 
            model_type=model_type.value["type"]
        )

        # If a model associated to the device already exists, delete it to replace it with
        # the new model, so that the system is always with the newest model in order to 
        # be constantly learning new network traffic patterns
        if old_run_id is not None:
            print(f"Old model from device {dev_addr} deleted.")
            
            self.__mlflowclient.delete_run(old_run_id)

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

        if old_model_id is not None:
            self.__mlflowclient.delete_logged_model(old_model_id)
            
            model_path = f"./mlruns/0/models/{old_model_id}"
            
            # If the path exists (which happens in normal cases), delete it
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                print(f"Model directory deleted: {model_path}")

            else:
                print(f"Model directory not found: {model_path}")
        
        # Create model based on DevAddr and store it as an artifact using MLFlow
        with mlflow.start_run(run_name=f'Model_Device_{dev_addr}_{dataset_type.value["name"].lower()}'):
            mlflow.set_tag("DevAddr", dev_addr)
            mlflow.set_tag("MessageType", dataset_type.value["name"].lower())
            
            ### Store the model as an artifact

            ## FOR sklearn MODELS (OCSVM, HBOS, LOF, sklearn-based IF)
            if model_type.value["type"] == "sklearn":
                mlflow.sklearn.log_model(sk_model=model, name="model")

            ## FOR spark models (kNN)
            elif model_type.value["type"] == "spark":
                mlflow.spark.log_model(spark_model=model, artifact_path="model")

            ## FOR pyod models (HBOS)
            elif model_type.value["type"] == "pyod":
                
                os.makedirs("./temp_models", exist_ok=True)    # Create the directory if it does not exist
                model_path = f'./temp_models/model_{dev_addr}_{dataset_type.value["name"].lower()}.pkl'

                # Save pkl model on original path
                with open(model_path, "wb") as f:
                    cloudpickle.dump(model, f)

                # Copy model artifact from original path to MLFlow
                mlflow.log_artifact(local_path=model_path, artifact_path="model")

                # Remove from original path
                os.remove(model_path)

            else:
                print("Model type must be sklearn, pyod or spark to be saved on MLFlow!")
                return      

            if os.path.exists(dataset_train_path):
                # Add dataset from original path to the MLFlow path
                mlflow.log_artifact(dataset_train_path, artifact_path="training_dataset")  

                # Remove dataset from original path
                shutil.rmtree(dataset_train_path)                                          
            else:
                print("Train dataset not found on original path")
            
            if os.path.exists(dataset_test_path):
                # Add dataset from original path to the MLFlow path
                mlflow.log_artifact(dataset_test_path, artifact_path="testing_dataset") 

                # Remove dataset from original path  
                shutil.rmtree(dataset_test_path)                                          
            else:
                print("Test dataset not found on original path")
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_dict(matrix, "confusion_matrix.json")
            mlflow.log_metric("recall_class_1", recall_anomalies)
            mlflow.log_dict(report, "report.json")

    """
    This function trains or re-trains a ML model based on a given DevAddr, and stores it on MLFlow
    It uses, as input, all samples of the pandas dataframe 'df' whose DevAddr is equal to 'dev_addr'
    This function is also used when the IDS receives a new message in real time and the model for the DevAddr
    of the message doesn't exist yet

    """
    def __create_model(self, df_model_train, df_model_test, dev_addr, dataset_type, datasets_format, model_type):

        start_time = time.time()

        ### LOF (Local Outlier Factor)
        if model_type.value["name"] == "Local Outlier Factor":      
            lof = LOF(df_train=df_model_train,
                  df_test=df_model_test,
                  featuresCol="features",
                  labelCol="intrusion")
        
            model = lof.train()
            accuracy, matrix, report = lof.test(model)
            recall_class_1 = report['1']['recall']

        ### kNN (k-Nearest Neighbors)
        elif model_type.value["name"] == "k-Nearest Neighbors":     
        
            knn = KNNAnomalyDetector(df_train=df_model_train,
                                    df_test=df_model_test,
                                    featuresCol="features",
                                    labelCol="intrusion",
                                    predictionCol="prediction")
            
            model = knn.train()
            accuracy, matrix, report = knn.test(model)
            recall_class_1 = report['1']['recall']

        ### HBOS (Histogram-Based Outlier Score)
        elif model_type.value["name"] == "Histogram-Based Outlier Score":
            hbos = HBOS(df_train=df_model_train, 
                    df_test=df_model_test,
                    featuresCol='features',
                    labelCol='intrusion')
        
            model = hbos.train()
            accuracy, matrix, report = hbos.test(model)
            recall_class_1 = report['1']['recall']

        ### Isolation Forest (sklearn)
        elif model_type.value["name"] == "Isolation Forest (Sklearn)":

            if_class = SkLearnIF(df_train=df_model_train, 
                                df_test=df_model_test, 
                                featuresCol="features",
                                labelCol="intrusion")
            
            model = if_class.train()
            accuracy, matrix, report = if_class.test(model)
            recall_class_1 = report['1']['recall']

        ### Isolation Forest (JAR from GitHub implementation)
        # NOTE: can't be saved on MLFlow
        elif model_type.value["name"] == "Isolation Forest (Custom)":
        
            if_class = CustomIF(spark_session=self.__spark_session,
                                df_train=df_model_train, 
                                df_test=df_model_test, 
                                featuresCol="features",
                                predictionCol="predictionCol",
                                labelCol="intrusion")
                                        
            model = if_class.train()
            matrix, report = if_class.test(model)
            accuracy = report["Accuracy"]
            recall_class_1 = report["Recall (class 1 -> anomaly)"]

        ### One-Class SVM
        elif model_type.value["name"] == "One-Class Support Vector Machine":

            ocsvm = OneClassSVM(spark_session=self.__spark_session,
                                df_train=df_model_train,
                                df_test=df_model_test,
                                featuresCol="features",
                                predictionCol="prediction",
                                labelCol="intrusion")
            
            model = ocsvm.train()
            matrix, report = ocsvm.test(model)
            accuracy = report["Accuracy"]
            recall_class_1 = report["Recall (class 1 -> anomaly)"]

        else:
            raise Exception("Model must correspond one of the algorithms used on the IDS!")
        
        if accuracy is not None:
            print(f'Accuracy for model of device {dev_addr} for {dataset_type.value["name"].upper()}: {round(accuracy * 100, 2)}%')
        
        if recall_class_1 is not None:
            print(f'Recall (class 1) for model of device {dev_addr} for {dataset_type.value["name"].upper()}: {round(recall_class_1 * 100, 2)}%')

        if matrix is not None:
            print("Confusion matrix:\n", matrix)
        
        if report is not None:
            print("Report:\n", json.dumps(report, indent=4))

        # NOTE uncomment the commented lines to store the test dataset
        # store the device datasets (training dataset will be used to be stored in MLFlow to be later retrieved for re-training)
        
        dataset_train_path = f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_{dev_addr}_train.{datasets_format}'
        dataset_test_path = f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_{dev_addr}_test.{datasets_format}'

        # Save final dataframe in JSON or PARQUET format
        if datasets_format == "json":
            df_model_train.coalesce(1).write.mode("overwrite").json(dataset_train_path)
            df_model_test.coalesce(1).write.mode("overwrite").json(dataset_test_path)
        else:
            df_model_train.coalesce(1).write.mode("overwrite").parquet(dataset_train_path)
            df_model_test.coalesce(1).write.mode("overwrite").parquet(dataset_test_path)

        # store the model on MLFlow
        self.__store_model(dev_addr, dataset_train_path, dataset_test_path, model, model_type, 
                           dataset_type, accuracy, matrix, recall_class_1, report)

        end_time = time.time()

        print(f'Model for end-device with DevAddr {dev_addr} and {dataset_type.value["name"].upper()} saved successfully and created in {format_time(end_time - start_time)}\n\n\n')

    """
    Function to create models for some of all devices

    It receives the spark session (spark_session) that handles the dataset processing and
    the corresponding dataset type (dataset_type) defined by DatasetType Enum

    It stores the models as artifacts using MLFlow, as well as their associated informations 
    such as metric evaluations and the associated DevAddr 

        dev_addr_list - a parameter to specify, as a list of strings, the DevAddr of the devices
                        from which the user pretends to create models

    """
    def create_ml_models(self, dev_addr_list, datasets_format):

        ### Begin processing
        start_time = time.time()

        if dev_addr_list is None:

            if datasets_format == "json":
                df_rxpk = self.__spark_session.read.json(f'./generatedDatasets/rxpk/lorawan_dataset.json')
                df_txpk = self.__spark_session.read.json(f'./generatedDatasets/txpk/lorawan_dataset.json')

            else:
                df_rxpk = self.__spark_session.read.parquet(f'./generatedDatasets/rxpk/lorawan_dataset.parquet')
                df_txpk = self.__spark_session.read.parquet(f'./generatedDatasets/txpk/lorawan_dataset.parquet')

            rxpk_counts = df_rxpk.groupBy("DevAddr").agg(count("*")).orderBy("DevAddr").collect()
            txpk_counts = df_txpk.groupBy("DevAddr").agg(count("*")).orderBy("DevAddr").collect()

            rxpk_devaddr_list = [str(row['DevAddr']) for row in rxpk_counts]
            txpk_devaddr_list = [str(row['DevAddr']) for row in txpk_counts]

            dev_addr_list = list(set(rxpk_devaddr_list + txpk_devaddr_list))

        model_type = ModelType.OCSVM

        # create each model in sequence
        for dev_addr in dev_addr_list:

            for dataset_type in DatasetType:

                # Pre-Processing phase
                df_model_train, df_model_test = prepare_df_for_device(
                    self.__spark_session, dataset_type, dev_addr, model_type, datasets_format
                )

                # If there are samples for the device, the model will be created
                if (df_model_train, df_model_test) != (None, None):
                    
                    # Processing phase
                    self.__create_model(df_model_train, df_model_test, dev_addr, 
                                        dataset_type, datasets_format, model_type)         
        
        end_time = time.time()

        # Print the total time; the time is in seconds, minutes or hours
        print("Total time of pre-processing + processing:", format_time(end_time - start_time), "\n\n")

    """
    Auxiliary function that classifies messages in real time, using the model that corresponds
    to the message's dev_addr

    TODO: add more parameters to the function if necessary
    
    """
    def classify_new_incoming_messages(self): 

        # TODO open sockets to listen LoRaWAN messages (RXPK and STATS)

        # TODO: review step 0
        # Read stream from kafka (e.g., port 9999)
        socket_stream_df = self.__spark_session.readStream \
                                .format("kafka") \
                                .option("host", "localhost") \
                                .option("port", 5200) \
                                .load()

        query = socket_stream_df.writeStream \
                .outputMode("append") \
                .format("console") \
                .start()

        query.awaitTermination()

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

