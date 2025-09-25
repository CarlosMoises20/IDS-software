
import time, mlflow, shutil, os, json, cloudpickle, threading, datetime
from zoneinfo import ZoneInfo
from pyspark.sql import Row
from pyspark.sql.types import IntegerType
import mlflow.sklearn
import pandas as pd
import numpy as np
from common.dataset_type import DatasetType
from prepareData.prepareData import *
from common.auxiliary_functions import *
from common.constants import KAFKA_PORT
from pyspark.sql.functions import count, regexp_extract
from mlflow.tracking import MlflowClient
from algorithms.one_class_svm import OneClassSVM
from algorithms.hbos import HBOS
from common.model_type import ModelType
from algorithms.lof import LOF
from algorithms.isolation_forest_custom import IsolationForest as CustomIF
from algorithms.isolation_forest_sklearn import IsolationForest as SkLearnIF
from prepareData.preProcessing.pre_processing import DataPreProcessing

"""
This class contains all necessary main functions to run the IDS in real time and create the models used to 
classify new incoming messages

"""
class MessageClassification:

    """
    The class is initialized with the following parameters
    
        spark_session: the Spark Session used for dataframe manipulation
        ml_algorithm: the object that represents the chosen Machine Learning algorithm to create the models
        mlflowclient: the client used to manipulate data on MLFlow, including stored models, datasets, tags, etc
        experiment_id: the ID of the experiment in MLFlow used to store all necessary data on MLFlow, since models, artifacts, tags, etc
    
    """
    def __init__(self, spark_session, ml_algorithm):
        self.__spark_session = spark_session
        self.__mlflowclient = MlflowClient()
        self.__experiment_id = self.__create_or_get_experiment()
        self.__ml_algorithm = next((m for m in ModelType if m.value["acronym"] == ml_algorithm), None)

    """
    Function to create an experiment on MLFlow if it does not exist yet
    This experiment will be used to store all necessary data, since models, artifacts, tags, etc
    
    """
    def __create_or_get_experiment(self):
        experiment_name = "IDS Project"

        # Verifies if experiment already exists
        experiment = self.__mlflowclient.get_experiment_by_name(experiment_name)

        # If an experiment already exists, return the ID of that same experiment, to avoid to create another experiment
        # and use unnecessary space on the disk
        if experiment:
            print(f"Experiment with ID {experiment.experiment_id}")
            return experiment.experiment_id
        
        # Creates experiment if it does not exist
        else:
            experiment_id = self.__mlflowclient.create_experiment(experiment_name)
            print(f"Created experiment with ID {experiment_id}")
            return experiment_id

    """
    This function returns the MLFlow model based on the associated DevAddr, received in the
    parameter

    It returns a tuple with the model itself as a MLFlow artifact, and the id of the corresponding
    run in MLFlow
    
    """
    def __get_model_from_mlflow(self, dev_addr, dataset_type, model_type):

        # get name of dataset type (example: DatasetType.RXPK -> "rxpk")
        dtype_name = dataset_type.value["name"]
        
        # Search the model according to DevAddr and MessageType
        runs = self.__mlflowclient.search_runs(
            experiment_ids=[self.__experiment_id],
            filter_string=f"tags.DevAddr = '{dev_addr}' and tags.MessageType = '{dtype_name}'",
            max_results=1
        )

        # If there are no runs, there is no model, so return a tuple with None objects
        if not runs:
            return None, None, None, None, None

        # get the ID of the retrieved run, if it exists
        run_id = runs[0].info.run_id

        # attribute None as default value of all models' objects
        model, model_id, scaler_model, pca_model, svd_matrix, train_columns = None, None, None, None, None, None
        
        try:

            local_path_columns = f"./mlruns/{self.__experiment_id}/{run_id}/artifacts/train_columns.json"

            train_columns = mlflow.artifacts.load_dict(local_path_columns)

            # retrieve the model of StandardScaler used on pre-processing for scaling. This model must always exist
            scaler_model = mlflow.spark.load_model(f"./mlruns/{self.__experiment_id}/{run_id}/artifacts/scaler_model")

            # for sklearn models
            if model_type.value["type"] == "sklearn":

                # Path where all sklearn models must be saved
                path = f"./mlruns/{self.__experiment_id}/{run_id}/outputs"
                
                # Return the ID of the model on the indicated path, if it exists
                model_id = [name for name in os.listdir(path) 
                        if os.path.isdir(os.path.join(path, name)) and name.startswith("m-")]
                
                model_id = model_id[0] if model_id else None

                model_path = f"./mlruns/{self.__experiment_id}/models/{model_id}/artifacts" if model_id else None

                # This if/else block ensures that the load_model does not throw exception if the model path does not exist 
                # Without this if/else block, the try/catch block would stop to run
                # on the load_model function and not loading the other models (PCA, SVD)
                # that might exist
                if model_path and os.path.exists(model_path):
                    model = mlflow.sklearn.load_model(model_path)
                else:
                    print("Model does not exist on path", model_path)
            
            # for pyod models (HBOS)
            elif model_type.value["type"] == "pyod":
                
                # Path where all pyod models must be saved
                model_path = f'./mlruns/{self.__experiment_id}/{run_id}/artifacts/model/model_{dev_addr}_{dtype_name}.pkl'
                
                # This if/else block ensures that the load_model does not throw exception if the model path does not exist 
                # Without this if/else block, the try/catch block would stop to run
                # on the load_model function and not loading the other models (PCA, SVD)
                # that might exist
                if os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        model = cloudpickle.load(f)
                else:
                    print("Model does not exist!")

            # Model where spark-based PCA model must be stored, if it exists
            pca_path = f"./mlruns/{self.__experiment_id}/{run_id}/artifacts/pca_model"
            
            # This if/else block ensures that the load_model does not throw exception if the model path does not exist 
            # Without this if/else block, the try/catch block would stop to run
            # on the load_model function and not loading the other models
            if os.path.exists(pca_path):
                pca_model = mlflow.spark.load_model(pca_path)
            else:
                print("PCA model does not exist")

            # Model where SVD matrix must be stored, if it exists
            svd_path = f"./mlruns/{self.__experiment_id}/{run_id}/artifacts/svd_model/svd_matrix.npy"

            # This if/else block ensures that the load_model does not throw exception if the model path does not exist 
            # Without this if/else block, the try/catch block would stop to run
            # on the load_model function and not printing the message to indicate the user that the model does not exist
            if os.path.exists(svd_path):
                svd_matrix = np.load(svd_path, allow_pickle=True)
            else:
                print("SVD model does not exist")
            
        except:
            pass

        # NOTE you can uncomment these lines if you want these prints
        print("model:", model)
        print("scaler model:", scaler_model)
        #print("PCA model:", pca_model)
        #print("SVD matrix:", svd_matrix)

        # Create a dictionary where all models used for pre-processing are stored, for easier manipulation
        transform_models = {"StdScaler": scaler_model, "PCA": pca_model, "SVD": svd_matrix}
        transform_models = None if all(value is None for value in transform_models.values()) else transform_models
        
        return model, transform_models, train_columns, model_id, run_id

    """
    This function returns the MLFlow training dataset based on the associated DevAddr, received in the
    parameter
    
    """
    def __load_train_dataset_from_mlflow(self, dev_addr, dataset_type, datasets_format):
        
        # get name of dataset type (example: DatasetType.RXPK -> "rxpk")
        dtype_name = dataset_type.value["name"]

        # Search the model according to DevAddr and MessageType
        runs = self.__mlflowclient.search_runs(
            experiment_ids=[self.__experiment_id],
            filter_string=f"tags.DevAddr = '{dev_addr}' and tags.MessageType = '{dtype_name}'",
            max_results=1
        )

        # If there are no runs, there is no model, so return a tuple with None objects
        if not runs:
            return None

        # get the ID of the retrieved run, if it exists
        run_id = runs[0].info.run_id

        # Get the path where the train dataset must be stored if it exists
        path = f'./mlruns/{self.__experiment_id}/{run_id}/artifacts/training_dataset/lorawan_dataset_{dev_addr}_train.{datasets_format}'

        # If the train dataset does not exist on the indicated path, return None
        if not os.path.exists(path):
            print("Train dataset does not exist")
            return None

        print("Train dataset found on path", path)

        # Load the train dataset according to its format (JSON or PARQUET)
        if datasets_format == "json":
            df = self.__spark_session.read.json(path)
        else:
            df = self.__spark_session.read.parquet(path)
        
        return df

    """
    This method logs training dataset on MLFlow according to DevAddr aht MessageType
    
        df: the dataframe corresponding to the dataset that will be stored on MLFlow
        path: the original path where the dataset will be stored first
        artifact_path: the final path where the dataset will be stored, using the old path 'path' to save it to this path
        datasets_format: the format chosen to store the dataset on MLFlow (JSON or PARQUET)
        dataset_type: the type of LoRaWAN messages in the dataset (RXPK and TXPK)
    
    """
    def __log_dataset_on_mlflow(self, df, path, artifact_path, datasets_format, dev_addr, dataset_type):

        # name of the type of LoRaWAN dataset (RXPK or TXPK)
        dtype_name = dataset_type.value["name"]

        # Search the model according to DevAddr and MessageType
        runs = self.__mlflowclient.search_runs(
            experiment_ids=[self.__experiment_id],
            filter_string=f"tags.DevAddr = '{dev_addr}' and tags.MessageType = '{dtype_name}'",
            max_results=1
        )

        # 'old_run_id' is the ID of the previous run if it already exists
        old_run_id = runs[0].info.run_id if runs else None

        # NOTE: you can comment this line if you don't want to print this
        print("run id:", old_run_id)

        # Save final dataframe in JSON or PARQUET format if the dataframe exists and contains rows
        if df and not df.isEmpty():
            df_to_save = df.drop("features")     
            if datasets_format == "json":
                df_to_save.write.mode("overwrite").json(path)
            else:
                df_to_save.write.mode("overwrite").parquet(path)
            
            # Create model based on DevAddr and store it as an artifact using MLFlow
            # If there is a run associated to DevAddr and MessageType (old_run_id), use that run instead of creating another one
            # This guarantees that if, on a specific device and message type, a run is already associated and has data inside it,
            # that data will not be ignored by creating another run associated to that device and message type
            with mlflow.start_run(run_name=f'Model_Device_{dev_addr}_{dtype_name}', 
                                  experiment_id=self.__experiment_id,
                                  run_id=old_run_id if old_run_id else None):

                print("Active run id:", mlflow.active_run().info.run_id)

                mlflow.set_tag("DevAddr", dev_addr)
                mlflow.set_tag("MessageType", dtype_name)

                print("Os path exists:", os.path.exists(path))

                # Add dataset from original path to the MLFlow path
                mlflow.log_artifact(path, artifact_path=artifact_path)  

                # Remove dataset from original path
                shutil.rmtree(path)

        else:
            return                                          

    """
    Auxiliary function that stores on MLFlow the model based on DevAddr, replacing the old model if it exists
    This model is used on training and testing, whether on only creating models based on past data, or for classifying
    new messages in real time, or for the re-training process
    
    """
    def __store_model(self, dev_addr, model, model_type, dataset_type, matrix, report, transform_models, df_model_train_columns):

        # Verify if a model associated to the device already exists. If so, return it;
        # otherwise, return None
        _, _, _, old_model_id, old_run_id = self.__get_model_from_mlflow(
            dev_addr=dev_addr, 
            dataset_type=dataset_type, 
            model_type=model_type
        )

        # name of the type of LoRaWAN dataset (RXPK or TXPK)
        dtype_name = dataset_type.value["name"]

        # If a model associated to the device already exists, delete it to replace it with
        # the new model, so that the system is always with the newest model in order to 
        # be constantly learning new network traffic patterns
        if old_model_id is not None:
            
            # The paths where the model should be saved
            model_paths = [f"./mlruns/{self.__experiment_id}/models/{old_model_id}",
                           f"./mlruns/{self.__experiment_id}/{old_run_id}/outputs/{old_model_id}"]
            
            # If the path exists (which happens in normal cases), delete it to save space on local machine (but not from MLFlow)
            for model_path in model_paths:
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                    print(f"Model directory deleted: {model_path}")

                else:
                    print(f"Model directory not found: {model_path}")
        
        # Create model based on DevAddr and store it as an artifact using MLFlow
        with mlflow.start_run(run_name=f'Model_Device_{dev_addr}_{dtype_name}', 
                              experiment_id=self.__experiment_id,
                              run_id=old_run_id if old_run_id else None):
            
            # Set tags that will identify the run on MLFlow: DevAddr (device address) and MessageType (RXPK or TXPK)
            mlflow.set_tag("DevAddr", dev_addr)
            mlflow.set_tag("MessageType", dtype_name)
            
            if transform_models["StdScaler"]:
                mlflow.spark.log_model(spark_model=transform_models["StdScaler"], artifact_path="scaler_model")

            # Log PCA model used for stream processing if it exists
            if transform_models["PCA"]:
                mlflow.spark.log_model(spark_model=transform_models["PCA"], artifact_path="pca_model")
            
            # Log SVD matrix used for stream processing, if it exists
            if transform_models["SVD"] is not None:
                svd_path = "svd_matrix.npy"
                np.save(svd_path, transform_models["SVD"])
                mlflow.log_artifact(local_path=svd_path, artifact_path="svd_model")
                os.remove(svd_path)
            
            ### Store the model as an artifact

            ## FOR sklearn MODELS (OCSVM, HBOS, LOF, sklearn-based IF)
            if model_type.value["type"] == "sklearn":
                mlflow.sklearn.log_model(sk_model=model, name="model")

            ## FOR pyod models (HBOS)
            elif model_type.value["type"] == "pyod":
                os.makedirs("./temp_models", exist_ok=True)    # Create the directory if it does not exist
                model_path = f'./temp_models/model_{dev_addr}_{dtype_name}.pkl'

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

            print("New model stored")

            mlflow.log_dict(df_model_train_columns, "train_columns.json")

            ### Log evaluation metrics on the MLFlow run
            if matrix:
                mlflow.log_dict(matrix, "confusion_matrix.json")
            
            if report:
                mlflow.log_dict(report, "report.json")

    """
    This function trains a ML model based on a given DevAddr and type of message (RXPK or TXPK), and stores it on MLFlow

    """
    def __create_model(self, df_model_train, df_model_test, dev_addr, dataset_type, model_type, transform_models):

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
            f1_score_class_1 = report['1']['f1-score']

        ### HBOS (Histogram-Based Outlier Score)
        elif model_type.value["name"] == "Histogram-Based Outlier Score":
            hbos = HBOS(df_train=df_model_train, 
                    df_test=df_model_test,
                    featuresCol='features',
                    labelCol='intrusion')
        
            model = hbos.train()
            accuracy, matrix, report = hbos.test(model)
            recall_class_1 = report['1']['recall']
            f1_score_class_1 = report['1']['f1-score']

        ### Isolation Forest (sklearn)
        elif model_type.value["name"] == "Isolation Forest (SkLearn)":

            if_class = SkLearnIF(df_train=df_model_train, 
                                df_test=df_model_test, 
                                featuresCol="features",
                                labelCol="intrusion")
            
            model = if_class.train()
            accuracy, matrix, report = if_class.test(model)
            recall_class_1 = report['1']['recall']
            f1_score_class_1 = report['1']['f1-score']

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
            f1_score_class_1 = report["F1-Score (class 1 -> anomaly)"]
            recall_class_1 = report["Recall (class 1 -> anomaly)"]

        ### One-Class SVM
        elif model_type.value["name"] == "One-Class Support Vector Machine":

            ocsvm = OneClassSVM(df_train=df_model_train,
                                df_test=df_model_test,
                                featuresCol="features",
                                labelCol="intrusion")
            
            model = ocsvm.train()
            accuracy, matrix, report = ocsvm.test(model)
            recall_class_1 = report['1']['recall']
            f1_score_class_1 = report['1']['f1-score']

        else:
            raise Exception("Model must correspond one of the algorithms used on the IDS!")
        
        ### Print evaluation metrics if they are not None

        if recall_class_1 is not None:
            print(f'Recall (class 1) for model of device {dev_addr} for {dataset_type.value["name"].upper()}: {round(recall_class_1 * 100, 2)}%')

        if f1_score_class_1 is not None:
            print(f'F1-Score (class 1) for model of device {dev_addr} for {dataset_type.value["name"].upper()}: {round(f1_score_class_1 * 100, 2)}%')

        if accuracy is not None:
            print(f'Accuracy for model of device {dev_addr} for {dataset_type.value["name"].upper()}: {round(accuracy * 100, 2)}%')

        if matrix is not None:
            print("Confusion matrix:\n", matrix)
        
        if report is not None:
            print("Report:\n", json.dumps(report, indent=4))

        # store the model on MLFlow
        self.__store_model(dev_addr=dev_addr, model=model, model_type=model_type, 
                           dataset_type=dataset_type, matrix=matrix, 
                           report=report, transform_models=transform_models, 
                           df_model_train_columns=df_model_train.columns)

        end_time = time.time()

        print(f'Model for end-device with DevAddr {dev_addr} and {dataset_type.value["name"].upper()} saved successfully and created in {format_time(end_time - start_time)}\n\n\n')

        return model, transform_models
    
    """
    This method is used for stream processing, to train the model with all dataset samples instead of splitting it
    into training and testing
    
    """
    def __train_model(self, df_model, dev_addr, dataset_type, model_type, transform_models):

        ### LOF (Local Outlier Factor)
        if model_type.value["name"] == "Local Outlier Factor":      
            lof = LOF(df_train=df_model,
                  df_test=None,
                  featuresCol="features",
                  labelCol="intrusion")
        
            model = lof.train()

        ### HBOS (Histogram-Based Outlier Score)
        elif model_type.value["name"] == "Histogram-Based Outlier Score":
            hbos = HBOS(df_train=df_model, 
                    df_test=None,
                    featuresCol='features',
                    labelCol='intrusion')
        
            model = hbos.train()

        ### Isolation Forest (sklearn)
        elif model_type.value["name"] == "Isolation Forest (Sklearn)":

            if_class = SkLearnIF(df_train=df_model, 
                                df_test=None, 
                                featuresCol="features",
                                labelCol="intrusion")
            
            model = if_class.train()

        ### Isolation Forest (JAR from GitHub implementation)
        ##### NOTE: can't be saved on MLFlow
        elif model_type.value["name"] == "Isolation Forest (Custom)":
        
            if_class = CustomIF(spark_session=self.__spark_session,
                                df_train=df_model, 
                                df_test=None, 
                                featuresCol="features",
                                predictionCol="predictionCol",
                                labelCol="intrusion")
                                        
            model = if_class.train()

        ### One-Class SVM
        elif model_type.value["name"] == "One-Class Support Vector Machine":

            ocsvm = OneClassSVM(df_train=df_model,
                                df_test=None,
                                featuresCol="features",
                                labelCol="intrusion")
            
            model = ocsvm.train()

        else:
            raise Exception("Model must correspond one of the algorithms used on the IDS!")
        
        # store the model on MLFlow
        self.__store_model(dev_addr=dev_addr, model=model, model_type=model_type, 
                           dataset_type=dataset_type, matrix=None, 
                           report=None, transform_models=transform_models, df_model_train_columns=df_model.columns)
        
        return model, transform_models

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

        # NOTE: you can comment this line if you don't want to print this
        print("Using algorithm", self.__ml_algorithm.value["name"])

        # If the user does not specify the list of DevAddr's manually, by default all available DevAddr's will be used
        # this returns a list of strings with all DevAddr's if it's the case
        if dev_addr_list is None:

            # For the case of retrieving the JSON dataset
            if datasets_format == "json":
                df_rxpk = self.__spark_session.read.json(f'./generatedDatasets/rxpk/lorawan_dataset.json')
                df_txpk = self.__spark_session.read.json(f'./generatedDatasets/txpk/lorawan_dataset.json')

            # For the case of retrieving the PARQUET dataset
            else:
                df_rxpk = self.__spark_session.read.parquet(f'./generatedDatasets/rxpk/lorawan_dataset.parquet')
                df_txpk = self.__spark_session.read.parquet(f'./generatedDatasets/txpk/lorawan_dataset.parquet')

            rxpk_list = df_rxpk.groupBy("DevAddr").agg(count("*")).orderBy("DevAddr").collect()
            txpk_list = df_txpk.groupBy("DevAddr").agg(count("*")).orderBy("DevAddr").collect()

            rxpk_devaddr_list = [str(row['DevAddr']) for row in rxpk_list]
            txpk_devaddr_list = [str(row['DevAddr']) for row in txpk_list]

            dev_addr_list = list(set(rxpk_devaddr_list + txpk_devaddr_list))

        # create each model in sequence
        for dev_addr in dev_addr_list:

            # loop for each DatasetType (RXPK and TXPK) inside each device
            for dataset_type in DatasetType:
                
                # For the case of retrieving the JSON dataset
                if datasets_format == "json":
                    df_model = self.__spark_session.read.json(
                        f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset.{datasets_format}'
                    ).filter(col("DevAddr") == dev_addr)

                # For the case of retrieving the PARQUET dataset
                else:
                    df_model = self.__spark_session.read.parquet(
                        f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset.{datasets_format}'
                    ).filter(col("DevAddr") == dev_addr)

                # Applies cache operations to speed up the processing
                df_model = df_model.cache()     

                # Start the process to create the model for the device, if there are enough samples for the device
                self.__model_creation_process(df_model=df_model,
                                              dataset_type=dataset_type,
                                              dev_addr=dev_addr, 
                                              model_type=self.__ml_algorithm, 
                                              datasets_format=datasets_format)
        
        end_time = time.time()

        # Print the total time; the time is in seconds, minutes or hours
        print("Total time of pre-processing + processing:", format_time(end_time - start_time), "\n\n")

    """
    This method represents the necessary process to create a model for a DevAddr and MessageType,
    since verifying if there are enough samples for the device, imputing missing values with the median 
    and create the model if there are enough samples for it

        df_model: the dataframe with the messages for the indicated device and message type
        dataset_type: type of messages on the dataset (RXPK or TXPK)
        dev_addr: the DevAddr corresponding to the model
        model_type: the ML algorithm that will be used to create the model
        datasets_format: the format of the datasets to be retrieved or generated (JSON or PARQUET)
        stream_processing (default=False): flag that indicates if messages are being received in real time (True) or if it's only
                                            to create models (False)

    """
    def __model_creation_process(self, df_model, dataset_type, dev_addr, model_type, datasets_format, stream_processing=False):

        # Pre-Processing phase
        df_model_train, df_model_test, transform_models = prepare_df_for_device(
            df_model=df_model,
            dataset_type=dataset_type, 
            dev_addr=dev_addr, 
            model_type=model_type,
            stream_processing=stream_processing
        )

        # If it's only to create models
        if not stream_processing:
            
            # If there are samples for the device, the model will be created, otherwise this step will be skipped
            if (df_model_test, transform_models) != (None, None):
                
                # Processing phase: create the model
                return self.__create_model(df_model_train, df_model_test, dev_addr, dataset_type, 
                                            model_type, transform_models) 

        # If receiving messages in real time
        else:

            # If there is at least one sample for the dataset, log the dataset on MLFlow and train the model with that dataset
            if df_model_train is not None:

                # training dataset will be used to be stored in MLFlow to be later retrieved for re-training
                self.__log_dataset_on_mlflow(df=df_model_train,
                                        path=f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_{dev_addr}_train.{datasets_format}', 
                                        artifact_path="training_dataset", 
                                        datasets_format=datasets_format,
                                        dev_addr=dev_addr,
                                        dataset_type=dataset_type)

                if transform_models is not None:
                
                    # train the model and return the created models, which include the models used for feature reduction  
                    return self.__train_model(df_model=df_model_train,
                                            dev_addr=dev_addr,
                                            dataset_type=dataset_type,
                                            model_type=model_type,
                                            transform_models=transform_models)
                
        # If no models are returned, return a tuple of None objects to indicate that no models were created
        return None, None        

    """
    Auxiliary function that classifies messages in real time, using the model that corresponds
    to the message's dev_addr

        datasets_format: the expected format of the used datasets (JSON or PARQUET)
    
    """
    def classify_new_incoming_messages(self, datasets_format): 

        # Start a thread for an asynchronous call to the function that will open a UDP socket that will receive messages
        # from a LoRa gateway and forward to a Kafka producer which will also be listening messages from UDP socket to
        # forward to Spark that will consume those messages
        threading.Thread(target=udp_to_kafka_forwarder, daemon=True).start()

        # NOTE: you can comment this line if you don't want to print this
        print("Using algorithm", self.__ml_algorithm.value["name"])

        # This function will be executed for each batch of one or more messages received by Spark
        def process_batch(df, batch_id):

            # TODO date is only for testing, remove it for final version to avoid confusions with timezone in different countries
            print(f'\n=== Batch with ID {batch_id}; Date: {datetime.datetime.now(ZoneInfo("Europe/Lisbon")).strftime("%d-%m-%Y %H:%M:%S")} (Lisbon Time) ===')

            df.show(truncate=False)

            # RXPK messages
            df_rxpk = df.filter(df.message.startswith('{"rxpk"'))

            # Only RXPK messages will be considered
            dataset_type = DatasetType.RXPK

            # Check if there are RXPK messages on the dataframe
            if not df_rxpk.isEmpty():
                
                print("RXPK Processing")

                # transform the dataframe before using the corresponding messages to classify them
                df_rxpk = pre_process_type(df=df_rxpk, dataset_type=dataset_type, stream_processing=True)

                # NOTE: this prints the message after it is pre-processed. You can comment this if you want
                print("Transformed messages")
                df_rxpk.show()

                # get DevAddr to use it to get the model from MLFlow
                dev_addrs = [str(row['DevAddr']) for row in df_rxpk.select("DevAddr").distinct().collect() if row['DevAddr'] is not None]

                # Loop for each DevAddr in the batch
                for dev_addr in dev_addrs:

                    print(f"------- Device {dev_addr} -------\n\n")

                    # Filter messages to only contain RXPK messages
                    df_device = df_rxpk.filter(col("DevAddr") == dev_addr)

                    print("Device stream df")
                    df_device.show(truncate=False)

                    # Retrieve the model using DevAddr
                    model, transform_models, stored_df_model_columns, _, _ = self.__get_model_from_mlflow(
                        dev_addr=dev_addr, dataset_type=dataset_type, model_type=self.__ml_algorithm
                    )

                    df_model = None

                    # If there is not previous model saved on MLFlow, try to create one
                    # If there is not enough samples to create the model, it won't be created, and
                    # the result will be a tuple of None objects 
                    if model is None:

                        # Get the training dataset that was saved on MLFlow, with samples from static dataset and also from gateway
                        df_model = self.__load_train_dataset_from_mlflow(dev_addr=dev_addr,
                                                                            dataset_type=dataset_type,
                                                                            datasets_format=datasets_format)
                        
                        # if there is no previous training dataset stored on MLFlow, retrieve from static datasets
                        if df_model is None:

                            print(f"Still no store training dataset for device {dev_addr} in RXPK. Creating the first from static datasets")

                            # for JSON format
                            if datasets_format == "json":
                                df_model = self.__spark_session.read.json(
                                    f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset.{datasets_format}'
                                ).filter(col("DevAddr") == dev_addr)

                            # for PARQUET format
                            else:
                                df_model = self.__spark_session.read.parquet(
                                    f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset.{datasets_format}'
                                ).filter(col("DevAddr") == dev_addr)                           

                        # try to create the model using the dataset which was loaded from MLFlow, or the dataset
                        # created using static datasets if they exist
                        # if static datasets do not exist or there are no enough samples to create the model,
                        # no model will be created
                        model, transform_models = self.__model_creation_process(df_model=df_model,
                                                                                dataset_type=dataset_type,
                                                                                dev_addr=dev_addr,
                                                                                model_type=self.__ml_algorithm,
                                                                                datasets_format=datasets_format,
                                                                                stream_processing=True)
                        
                        if model:
                            stored_df_model_columns = df_model.columns

                        # Applies cache operations to speed up the dataframe processing
                        df_model = df_model.cache()

                        # NOTE: this prints from MLFlow or the static dataset corresponding to DevAddr. You can comment this if you want
                        n_model_samples = df_model.count()

                        if n_model_samples == 1:
                            print("Loaded dataset or static dataset, with", n_model_samples, "sample")
                        else:
                            print("Loaded dataset or static dataset, with", n_model_samples, "samples")
                            
                        df_model.show(truncate=False)


                    # If the models are created, use the actual sample to train the model
                    if model and transform_models and stored_df_model_columns:

                        # Remove columns from the string list that are not used for machine learning
                        non_null_columns_device = [
                            c for c in df_device.columns
                            if (
                                # Check if NOT all values are null
                                (df_device.agg(sum(when(col(c).isNotNull(), 1).otherwise(0))).first()[0] or 0) > 0
                            ) and (c not in ["DevAddr"])
                        ]

                        columns_names = list(set(stored_df_model_columns + non_null_columns_device))

                        # NOTE uncomment if necessary
                        #df_bind = df_model.unionByName(df_device, allowMissingColumns=True)
                        
                        # if there is a column on 'df_device' that is missing that is not missing in the model dataset, and if
                        # the model exists, that NULL column of df_device is replaced with the mean according to df_model
                        imputer = Imputer(inputCols=columns_names, outputCols=columns_names, strategy="mean")
                        imputer_model = imputer.fit(df_model)
                        df_device = imputer_model.transform(df_device)

                        if columns_names != stored_df_model_columns:

                            # Assemble the attributes into "features" using the transform (Scaler, PCA and SVD) models from the device
                            # to transform the features of the dataframe
                            df_device = DataPreProcessing.features_assembler_stream(df=df_device,
                                                                                    df_model=df_model,
                                                                                    columns_names=columns_names,
                                                                                    model_type=self.__ml_algorithm,
                                                                                    transform_models=transform_models,
                                                                                    new_schema=True)
                        else:

                            df_device = DataPreProcessing.features_assembler_stream(df=df_device,
                                                                                    df_model=df_model,
                                                                                    columns_names=columns_names,
                                                                                    model_type=self.__ml_algorithm,
                                                                                    transform_models=transform_models)

                        # Convert into rows to join dataframes with its predictions
                        original_rows = df_device.collect()
                        
                        print("Device stream df after pre-processing")
                        
                        df_device.show(truncate=False)
                        
                        # Process "features" into a numpy array, the requested format for ML algorithm training
                        features = np.array(df_device.select("features").rdd.map(lambda x: x[0]).collect())
                        
                        # Apply model prediction on "features"
                        y_pred = model.predict(features)

                        print("Predictions:", y_pred)

                        # Bind messages with corresponding predictions
                        rows_with_preds = [Row(**row.asDict(), prediction=int(pred)) for row, pred in zip(original_rows, y_pred)]
                        
                        predictions = np.array([0 if pred == 1 else 1 for pred in y_pred])
                        
                        # NOTE: This prints the number of anomalies and normal messages detected, which is not recommended to uncomment
                        print(f"Number of anomalies detected: {(predictions == 1).sum()}")
                        print(f"Number of normal messages: {(predictions == 0).sum()}")

                        schema = df_device.schema.add(StructField('prediction', IntegerType(), False))

                        df_with_preds = self.__spark_session.createDataFrame(rows_with_preds, schema=schema)

                        # Create dataframe with new dataframes
                        df_normals = df_with_preds.filter(df_with_preds.prediction == 0)

                        print("Normal messages")
                        df_normals.show(truncate=False)

                        # Retrain the model with normal instances if they exist, and then store the new model on MLFlow
                        if not df_normals.isEmpty():
                            print("Re-training model")
                            features = np.array(df_normals.select("features").rdd.map(lambda x: x[0]).collect())
                            model = model.fit(features)

                            # fit transform models too
                            transform_models = DataPreProcessing.fit_transform_models_on_stream(df_normals=df_normals,
                                                                                                model_type=self.__ml_algorithm,
                                                                                                transform_models=transform_models)

                            self.__store_model(dev_addr=dev_addr, model=model,
                                            model_type=self.__ml_algorithm,
                                            dataset_type=dataset_type,
                                            matrix=None, report=None,
                                            transform_models=transform_models,
                                            df_model_train_columns=df_model.columns)
                            
                        df_device = df_normals
                        
                    # If no model was created for having no enough train samples, concatenate the new sample with the
                    # device training dataset which will be used later to create the first device model
                    if model is None:
                    
                        # bind the actual samples with the samples to update the training dataset, so that
                        # it can be used when necessary, for example, to retraining       
                        print("Binding training dataset to update it on MLFlow")
                    
                        df_result = df_device

                        # If a previous static dataset exists, concatenate the actual dataframe with the device samples
                        # from the static dataset, to form the dataframe to create                                      
                        if df_model:
                            df_result = df_result.unionByName(df_model, allowMissingColumns=True)

                        # NOTE: Print the number of samples of the training dataset to be logged on MLFLow. You can comment this 5 lines if you want
                        n_samples = df_result.count()
                        if n_samples == 1:
                            print(f"Logging dataset on MLFlow with {n_samples} sample")
                        else:
                            print(f"Logging dataset on MLFlow with {n_samples} samples")
                        
                        # Log the dataset on MLFlow to be used to retrain the model when a message with the same DevAddr is used
                        self.__log_dataset_on_mlflow(df=df_result,
                                        path=f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_{dev_addr}_train.{datasets_format}', 
                                        artifact_path="training_dataset", 
                                        datasets_format=datasets_format,
                                        dev_addr=dev_addr,
                                        dataset_type=dataset_type)
            
            # if there are no RXPK messages, skip the batch processing and keep listening new messages
            else:
                print("Batch with no RXPK messages.")
                return

        # Read stream from Kafka server that listens messages from UDP server
        socket_stream_df = self.__spark_session.readStream \
                                .format("kafka") \
                                .option("kafka.bootstrap.servers", f"localhost:{KAFKA_PORT}") \
                                .option("subscribe", "lorawan-messages") \
                                .load()

        # Convert bytes value to string
        decoded_df = socket_stream_df.selectExpr("CAST(value AS STRING) as message")

        # Extract the RXPK or STAT message as the corresponding JSON inside the message content, using a regex pattern
        # using the corresponding spark operation
        cleaned_df = decoded_df.withColumn(
            "message", regexp_extract("message", r'(\{"(?:rxpk|stat)".*)', 1)
        )

        # forEachBatch allows micro-batch processing, taking advantage of Spark operations to ensure data consistency
        # on each group of data, making it easier to capture fails on a group of data; it provides higher performance
        # for batch calls; and treats each entry as a Spark dataframe; batch processing is the best approach
        # to handle processing of large quantities of static data and creating heavy ML models
        query = cleaned_df.writeStream \
            .foreachBatch(process_batch) \
            .outputMode("append") \
            .start()
        
        # Wait until the user stops the script execution (Ctrl + C in Windows)
        query.awaitTermination()

