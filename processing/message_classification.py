
import time, mlflow, shutil, os, json, cloudpickle, threading

import mlflow.artifacts
from pyspark.sql import Row
import mlflow.sklearn
import numpy as np
from common.dataset_type import DatasetType
from prepareData.prepareData import *
from common.auxiliary_functions import format_time, udp_to_kafka_forwarder
from pyspark.sql.functions import count, regexp_extract
from mlflow.tracking import MlflowClient
from models.one_class_svm import OneClassSVM
from models.hbos import HBOS
from common.model_type import ModelType
from models.lof import LOF
from models.kNN import KNNAnomalyDetector
from models.isolation_forest_custom import IsolationForest as CustomIF
from models.isolation_forest_sklearn import IsolationForest as SkLearnIF
from prepareData.preProcessing.pre_processing import DataPreProcessing

class MessageClassification:

    def __init__(self, spark_session):
        self.__spark_session = spark_session
        self.__mlflowclient = MlflowClient()
        self.__experiment_id = self.__create_experiment()
        self.__model_type = ModelType.OCSVM

    """
    Function to create an experiment on MLFlow if it does not exist yet
    This experiment will be used to store all necessary data, since models, artifacts, tags, etc
    
    """
    def __create_experiment(self):
        experiment_name = "IDS Project"

        # Verifies if experiment already exists
        experiment = self.__mlflowclient.get_experiment_by_name(experiment_name)

        if experiment:
            print(f"Experiment with ID {experiment.experiment_id}")
            return experiment.experiment_id
        
        else:
            # Creates experiment if it does not exist
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
        
        # Search the model according to DevAddr and MessageType
        runs = self.__mlflowclient.search_runs(
            experiment_ids=[self.__experiment_id],
            filter_string=f"tags.DevAddr = '{dev_addr}' and tags.MessageType = '{dataset_type}'",
            max_results=1
        )
        
        if not runs:
            return None, None, None, None

        # for all
        run_id = runs[0].info.run_id
        model, model_id, scaler_model, pca_model, svd_matrix = None, None, None, None, None
        
        try:

            scaler_model = mlflow.spark.load_model(f"./mlruns/{self.__experiment_id}/{run_id}/artifacts/scaler_model")

            # for sklearn models
            if model_type.value["type"] == "sklearn":
                path = f"./mlruns/{self.__experiment_id}/{run_id}/outputs"
                model_id = [name for name in os.listdir(path) 
                        if os.path.isdir(os.path.join(path, name)) and name.startswith("m-")]
                model_id = model_id[0] if model_id else None

                model = mlflow.sklearn.load_model(f"./mlruns/{self.__experiment_id}/models/{model_id}/artifacts")
            
            # for spark models (kNN)
            elif model_type.value["type"] == "spark":
                model = mlflow.spark.load_model(f"./mlruns/{self.__experiment_id}/{run_id}/artifacts/model")

            # for pyod models (HBOS)
            elif model_type.value["type"] == "pyod":
                model_uri = f'./mlruns/{self.__experiment_id}/{run_id}/artifacts/model/model_{dev_addr}_{dataset_type}.pkl'
                with open(model_uri, "rb") as f:
                    model = cloudpickle.load(f)

            pca_path = f"./mlruns/{self.__experiment_id}/{run_id}/artifacts/pca_model"
            
            if os.path.exists(pca_path):
                pca_model = mlflow.spark.load_model(pca_path)

            svd_path = f"./mlruns/{self.__experiment_id}/{run_id}/artifacts/svd_model/svd_matrix.npy"

            if os.path.exists(svd_path):
                svd_matrix = np.load(svd_path, allow_pickle=True)
            
        except:
            pass

        # NOTE you can uncomment these lines if you want these prints
        #print("model:", model)
        #print("scaler model:", scaler_model)
        #print("PCA model:", pca_model)
        #print("SVD matrix:", svd_matrix)
        
        return model, {"StdScaler": scaler_model, "PCA": pca_model, "SVD": svd_matrix}, model_id, run_id


    """
    This function returns the MLFlow training dataset based on the associated DevAddr, received in the
    parameter
    
    """
    def __load_train_dataset_from_mlflow(self, dev_addr, dataset_type, datasets_format):
        
        # Search the model according to DevAddr and MessageType
        runs = self.__mlflowclient.search_runs(
            experiment_ids=[self.__experiment_id],
            filter_string=f"tags.DevAddr = '{dev_addr}' and tags.MessageType = '{dataset_type}'",
            max_results=1
        )
        
        if not runs:
            return None

        path = f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_{dev_addr}_train.{datasets_format}'

        if not os.path.exists(path):
            return None

        if datasets_format == "json":
            df = self.__spark_session.read.json(path)
        else:
            df = self.__spark_session.read.parquet(path)
        
        return df

    def __log_dataset_on_mlflow(self, df, path, artifact_path, datasets_format, dev_addr, dataset_type):

        # Search the model according to DevAddr and MessageType
        runs = self.__mlflowclient.search_runs(
            experiment_ids=[self.__experiment_id],
            filter_string=f"tags.DevAddr = '{dev_addr}' and tags.MessageType = '{dataset_type}'",
            max_results=1
        )

        old_run_id = runs[0].info.run_id

        if df and not df.isEmpty():
            # Save final dataframe in JSON or PARQUET format
            if datasets_format == "json":
                df.coalesce(1).write.mode("overwrite").json(path)
            else:
                df.coalesce(1).write.mode("overwrite").parquet(path)
            
            # Create model based on DevAddr and store it as an artifact using MLFlow
            with mlflow.start_run(run_name=f'Model_Device_{dev_addr}_{dataset_type.value["name"].lower()}', 
                                  experiment_id=self.__experiment_id,
                                  run_id=old_run_id if old_run_id else None):

                mlflow.set_tag("DevAddr", dev_addr)
                mlflow.set_tag("MessageType", dataset_type.value["name"].lower())

                # Add dataset from original path to the MLFlow path
                mlflow.log_artifact(path, artifact_path=artifact_path)  

                # Remove dataset from original path
                shutil.rmtree(path)                                          

    """
    Auxiliary function that stores on MLFlow the model based on DevAddr, replacing the old model if it exists
    This model is used on training and testing, whether on only creating models based on past data, or for classifying
    new messages in real time, or for the re-training process
    
    """
    def __store_model(self, dev_addr, model, model_type,
                      dataset_type, accuracy, matrix, recall_anomalies, 
                      report, transform_models):

        # Verify if a model associated to the device already exists. If so, return it;
        # otherwise, return None
        _, _, old_model_id, old_run_id = self.__get_model_from_mlflow(
            dev_addr=dev_addr, 
            dataset_type=dataset_type.value["name"].lower(), 
            model_type=model_type
        )

        # If a model associated to the device already exists, delete it to replace it with
        # the new model, so that the system is always with the newest model in order to 
        # be constantly learning new network traffic patterns
        if old_model_id is not None:
            self.__mlflowclient.delete_logged_model(old_model_id)
            
            model_path = f"./mlruns/{self.__experiment_id}/models/{old_model_id}"
            
            # If the path exists (which happens in normal cases), delete it
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                print(f"Model directory deleted: {model_path}")

            else:
                print(f"Model directory not found: {model_path}")
        
        # Create model based on DevAddr and store it as an artifact using MLFlow
        with mlflow.start_run(run_name=f'Model_Device_{dev_addr}_{dataset_type.value["name"].lower()}', 
                              experiment_id=self.__experiment_id,
                              run_id=old_run_id if old_run_id else None):
            
            mlflow.set_tag("DevAddr", dev_addr)
            mlflow.set_tag("MessageType", dataset_type.value["name"].lower())
            mlflow.spark.log_model(spark_model=transform_models["StdScaler"], artifact_path="scaler_model")

            if transform_models["PCA"]:
                mlflow.spark.log_model(spark_model=transform_models["PCA"], artifact_path="pca_model")
            
            if transform_models["SVD"] is not None:
                svd_path = "svd_matrix.npy"
                np.save(svd_path, transform_models["SVD"])
                mlflow.log_artifact(local_path=svd_path, artifact_path="svd_model")
                os.remove(svd_path)
            
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
            
            if accuracy:
                mlflow.log_metric("accuracy", accuracy)
            
            if matrix:
                mlflow.log_dict(matrix, "confusion_matrix.json")
            
            if recall_anomalies:
                mlflow.log_metric("recall_class_1", recall_anomalies)
            
            if report:
                mlflow.log_dict(report, "report.json")

    """
    This function trains or re-trains a ML model based on a given DevAddr, and stores it on MLFlow
    It uses, as input, all samples of the pandas dataframe 'df' whose DevAddr is equal to 'dev_addr'
    This function is also used when the IDS receives a new message in real time and the model for the DevAddr
    of the message doesn't exist yet

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

        # store the model on MLFlow
        self.__store_model(dev_addr, model, model_type, 
                           dataset_type, accuracy, matrix, recall_class_1, 
                           report, transform_models)

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

        ### kNN (k-Nearest Neighbors)
        elif model_type.value["name"] == "k-Nearest Neighbors":     
        
            knn = KNNAnomalyDetector(df_train=df_model,
                                    df_test=None,
                                    featuresCol="features",
                                    labelCol="intrusion",
                                    predictionCol="prediction")
            
            model = knn.train()

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
        # NOTE: can't be saved on MLFlow
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

            ocsvm = OneClassSVM(spark_session=self.__spark_session,
                                df_train=df_model,
                                df_test=None,
                                featuresCol="features",
                                predictionCol="prediction",
                                labelCol="intrusion")
            
            model = ocsvm.train()

        else:
            raise Exception("Model must correspond one of the algorithms used on the IDS!")
        
        # store the model on MLFlow
        self.__store_model(dev_addr=dev_addr, model=model, model_type=model_type, 
                           dataset_type=dataset_type, accuracy=None, matrix=None, recall_class_1=None, 
                           report=None, transform_models=transform_models)
        
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

        # create each model in sequence
        for dev_addr in dev_addr_list:
            for dataset_type in DatasetType:

                if datasets_format == "json":
                    df_model = self.__spark_session.read.json(
                        f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset.{datasets_format}'
                    ).filter(col("DevAddr") == dev_addr)

                else:
                    df_model = self.__spark_session.read.parquet(
                        f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset.{datasets_format}'
                    ).filter(col("DevAddr") == dev_addr)

                df_model = df_model.cache()     # Applies cache operations to speed up the processing

                self.__model_creation_process(df_model=df_model,
                                              dataset_type=dataset_type,
                                              dev_addr=dev_addr, 
                                              model_type=self.__model_type, 
                                              datasets_format=datasets_format)
        
        end_time = time.time()

        # Print the total time; the time is in seconds, minutes or hours
        print("Total time of pre-processing + processing:", format_time(end_time - start_time), "\n\n")

    def __model_creation_process(self, df_model, dataset_type, dev_addr, model_type, datasets_format, stream_processing=False):

        # Pre-Processing phase
        df_model_train, df_model_test, transform_models = prepare_df_for_device(
            df_model=df_model,
            dataset_type=dataset_type, 
            dev_addr=dev_addr, 
            model_type=model_type,
            stream_processing=stream_processing
        )

        if not stream_processing:
            
            # If there are samples for the device, the model will be created
            if (df_model_test, transform_models) != (None, None):
                
                # Processing phase
                return self.__create_model(df_model_train, df_model_test, dev_addr, dataset_type, 
                                            model_type, transform_models) 

        else:

            if df_model_train is not None:

                # training dataset will be used to be stored in MLFlow to be later retrieved for re-training
                self.__log_dataset_on_mlflow(df=df_model_train,
                                        path=f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_{dev_addr}_train.{datasets_format}', 
                                        artifact_path="training_dataset", 
                                        datasets_format=datasets_format,
                                        dev_addr=dev_addr,
                                        dataset_type=DatasetType.RXPK)
                
                return self.__train_model(df_model=df_model_train,
                                        dev_addr=dev_addr,
                                        dataset_type=dataset_type,
                                        model_type=model_type,
                                        transform_models=transform_models)
                

        return None, None        


    """
    Auxiliary function that classifies messages in real time, using the model that corresponds
    to the message's dev_addr

    TODO: add more parameters to the function if necessary
    
    """
    def classify_new_incoming_messages(self, datasets_format): 

        threading.Thread(target=udp_to_kafka_forwarder, daemon=True).start()

        def process_batch(df, batch_id):

            print(f"\n=== Batch with ID {batch_id} ===")

            df.show(truncate=False)

            # RXPK messages
            df_rxpk = df.filter(df.message.startswith('{"rxpk"'))

            if not df_rxpk.isEmpty():
                
                print("RXPK Processing")

                df_rxpk = pre_process_type(df=df_rxpk, dataset_type=DatasetType.RXPK, stream_processing=True)

                # get DevAddr to use it to get the model from MLFlow
                dev_addrs = [str(row['DevAddr']) for row in df_rxpk.select("DevAddr").distinct().collect()]

                # Loop for each DevAddr in the batch
                for dev_addr in dev_addrs:

                    print(f"------- Device {dev_addr} -------\n\n")

                    df_device = df_rxpk.filter(col("DevAddr") == dev_addr)

                    # Retrieve the model using DevAddr
                    model, transform_models, _, _ = self.__get_model_from_mlflow(
                        dev_addr=dev_addr, dataset_type=DatasetType.RXPK, model_type=self.__model_type
                    )

                    # If there is not previous model saved on MLFlow, try to create one
                    # If there is not enough samples to create the model, it won't be created, and
                    # the result will be a tuple of None objects 
                    if (model, transform_models) == (None, None):

                        print("Still no model created for device in RXPK. Creating the first")

                        # Get the training dataset that was saved on MLFlow
                        df_model = self.__load_train_dataset_from_mlflow(dev_addr=dev_addr,
                                                                            dataset_type=DatasetType.RXPK,
                                                                            datasets_format=datasets_format)
                        
                        # If there is no previous training dataset from MLFlow, create one from static datasets
                        if df_model is None:

                            print("Creating model from static dataset")

                            if datasets_format == "json":
                                df_model = self.__spark_session.read.json(
                                    f'./generatedDatasets/{DatasetType.RXPK.value["name"]}/lorawan_dataset.{datasets_format}'
                                ).filter(col("DevAddr") == dev_addr)

                            else:
                                df_model = self.__spark_session.read.parquet(
                                    f'./generatedDatasets/{DatasetType.RXPK.value["name"]}/lorawan_dataset.{datasets_format}'
                                ).filter(col("DevAddr") == dev_addr)

                        df_model = df_model.cache()     # Applies cache operations to speed up the processing

                        model, transform_models = self.__model_creation_process(df_model=df_model,
                                                                                dataset_type=DatasetType.RXPK,
                                                                                dev_addr=dev_addr,
                                                                                model_type=self.__model_type,
                                                                                datasets_format=datasets_format,
                                                                                stream_processing=True)

                    # If the model is created, use the actual sample to train the model
                    if (model, transform_models) != (None, None):

                        original_rows = df_device.collect()

                        # Assemble the attributes into "features" using the transform (Scaler, PCA and SVD) models from the device
                        # to transform the features of the dataframe
                        df_device = DataPreProcessing.features_assembler_stream(df=df_device,
                                                                                model_type=self.__model_type,
                                                                                transform_models=transform_models)
                        
                        features = np.array(df_device.select("features").rdd.map(lambda x: x[0]).collect())
                        
                        y_pred = model.predict(features)

                        rows_with_preds = [Row(**row.asDict(), prediction=int(pred)) for row, pred in zip(original_rows, y_pred)]

                        predictions = [np.array([0 if pred == 1 else 1 for pred in y_pred])]

                        print(f"Number of anomalies detected: {predictions.count(1)}")
                        print(f"Number of normal messages: {predictions.count(0)}")

                        df_with_preds = self.__spark_session.createDataFrame(rows_with_preds)
                        df_normals = df_with_preds.filter(df_with_preds.prediction == 0)

                        print("Normal messages")
                        df_normals.show(truncate=False)

                        # Retrain the model with the normal instances
                        features = np.array(df_normals.select("features").rdd.map(lambda x: x[0]).collect())
                        model = model.fit(features)

                        df_device = df_normals

                        self.__store_model(dev_addr=dev_addr, model=model,
                                        model_type=self.__model_type,
                                        dataset_type=DatasetType.RXPK,
                                        accuracy=None, matrix=None, recall_anomalies=None, report=None,
                                        transform_models=transform_models)
                        
                    # if the model is not created for not having enough samples for it, bind the actual samples with the samples
                    # from the training dataset on MLFlow, to make it to the minimum number of necessary samples to create the first model
                    else:

                        print("Not enough samples. Binding until it reaches the minimum limit")

                        # Get the training dataset that was saved on MLFlow
                        df_model_train = self.__load_train_dataset_from_mlflow(dev_addr=dev_addr,
                                                                            dataset_type=DatasetType.RXPK,
                                                                            datasets_format=datasets_format)
                    
                        # If a previous static dataset exists, concatenate the actual dataframe with the device samples
                        # from the static dataset, to form the dataframe to create                                      
                        if df_model_train is not None:
                            df_device = df_device.unionByName(df_model_train, allowMissingColumns=True)

                    print("Logging dataset on MLFlow")
                    
                    self.__log_dataset_on_mlflow(df=df_device,
                                     path=f'./generatedDatasets/{DatasetType.RXPK.value["name"]}/lorawan_dataset_{dev_addr}_train.{datasets_format}', 
                                     artifact_path="training_dataset", 
                                     datasets_format=datasets_format,
                                    dev_addr=dev_addr,
                                    dataset_type=DatasetType.RXPK)
            
            else:
                print("Batch with no RXPK messages.")
                return

        # Read stream from Kafka server that listens messages from UDP server
        socket_stream_df = self.__spark_session.readStream \
                                .format("kafka") \
                                .option("kafka.bootstrap.servers", "localhost:9092") \
                                .option("subscribe", "lorawan-messages") \
                                .load()

         # Convert bytes value to string
        decoded_df = socket_stream_df.selectExpr("CAST(value AS STRING) as message")

        cleaned_df = decoded_df.withColumn(
            "message", regexp_extract("message", r'(\{"(?:rxpk|stat|txpk)".*)', 1)
        )

        # forEachBatch allows micro-batch processing, taking advantage of Spark operations to ensure data consistency
        # on each group of data, making it easier to capture fails on a group of data; it provides higher performance
        # for batch calls; and treats each entry as a Spark dataframe; batch processing is the best approach
        # to handle processing of large quantities of static data and creating heavy ML models
        query = cleaned_df.writeStream \
            .foreachBatch(process_batch) \
            .outputMode("append") \
            .start()
        
        query.awaitTermination()

