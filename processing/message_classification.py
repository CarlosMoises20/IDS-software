
import time, mlflow, shutil, os, json
import mlflow.sklearn
from common.dataset_type import DatasetType
from prepareData.prepareData import prepare_df_for_device
from common.auxiliary_functions import format_time
from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.cluster import DBSCAN
from models.one_class_svm import OneClassSVM
from models.hbos import HBOS
from models.lof import LOF
from models.kNN import *
from pyspark.sql.streaming import DataStreamReader
from models.autoencoder import Autoencoder
from models.isolation_forest import *

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
            # TODO review this for the case of sklearn, pyod and pytorch models!
            #model = mlflow.spark.load_model(f"./mlruns/0/{run_id}/artifacts/model")
            model = mlflow.sklearn.load_model(f"./mlruns/0/{run_id}/artifacts/model")
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
        # TODO fix this to also replace the model in case of sklearn, pyod and pytorch models!
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
        # TODO consider a generated ID to represent the run id for the logic specially 
        # for the case of sklearn, pyod and pytorch models!
        with mlflow.start_run(run_name=f'Model_Device_{dev_addr}_{dataset_type.value["name"].lower()}'):
            mlflow.set_tag("DevAddr", dev_addr)
            mlflow.set_tag("MessageType", dataset_type.value["name"].lower())
            
            # for every algorithms except kNN, store the model as artifact
            if model is not None:

                ## FOR ISOLATION FOREST
                ## ??

                ## FOR THE OTHER MODELS
                mlflow.sklearn.log_model(model, "model") 
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
    def __create_model(self, df_model_train, df_model_test, dev_addr, dataset_type, datasets_format):

        start_time = time.time()

        """# LOF (Local Outlier Factor)

        lof = LOF(df_train=df_model_train,
                  df_test=df_model_test,
                  featuresCol="features",
                  labelCol="intrusion")
        
        #model = lof.train()

        #accuracy, matrix, report = lof.test(model)

        accuracy_list = []
        class_1_recall_list = []
        class_0_recall_list = []
        models = []
        matrix_list = []
        report_list = []
        n_neighbors_list = [3, 5, 7, 10]

        # test models with different k levels of the training dataset, to return the model with the best results
        for n_neighbors in n_neighbors_list:
            model = lof.train(n_neighbors=n_neighbors)
            accuracy, matrix, report = lof.test(model)
            accuracy_list.append(accuracy)
            class_1_recall_list.append(report['1']['recall'])
            class_0_recall_list.append(report['0']['recall'])
            models.append(model)
            matrix_list.append(matrix)
            report_list.append(report)

            # NOTE: uncomment these lines to print the results for each k-case model
            #print(f'Recall (class 1) for k={n_neighbors}: {round(report["1"]["recall"] * 100, 2)}%')
            #print(f'Recall (class 0) for k={n_neighbors}: {round(report["0"]["recall"] * 100, 2)}%')
            #print(f"CM for k={n_neighbors}:\n{matrix}")
            #print(f"Accuracy for k={n_neighbors}: {round(accuracy * 100, 2)}%\n")
            #print(f"Report for k={n_neighbors} and {n_bins} bins:\n{json.dumps(report, indent=4)}\n\n\n")

        # return the highest recall for class 1 (to ensure that the model detects as much anomalies as possible)
        max_class_1_recall = max(class_1_recall_list)
        indices_max_recall_1  = [i for i, r in enumerate(class_1_recall_list) if r == max_class_1_recall]
        best_index = indices_max_recall_1[0]
        min_recall_0 = class_0_recall_list[best_index]

        # if two models or more have the same highest recall for class 1, choose the model with the lowest recall for class 0
        # to ensure that we detect as much anomalies as possible. It's better to have some false positives than false negatives
        # We want to minimize false negatives (anomalies detected as normal packets) as much as possible
        # If the recall for class 0 is still the same between those models, choosing one model or another is the same thing,
        # since the accuracy is the same because the test dataset is the same
        for i in indices_max_recall_1[1:]:
            if class_0_recall_list[i] < min_recall_0:
                best_index = i
                min_recall_0 = class_0_recall_list[i]

        recall_class_1 = class_1_recall_list[best_index]
        #class_0_recall = class_0_recall_list[best_index]
        print(f"Chosen k: {n_neighbors_list[best_index]}")
        accuracy = accuracy_list[best_index]
        model = models[best_index]
        report = report_list[best_index]"""

        """#kNN (k-Nearest Neighbors)
        
        knn = KNNAnomalyDetector(df_train=df_model_train,
                                df_test=df_model_test,
                                featuresCol="features",
                                labelCol="intrusion",
                                predictionCol="prediction")
        
        model = knn.train()
        accuracy, matrix, report = knn.test(model)"""

        """# HBOS (Histogram-Based Outlier Score)
        hbos = HBOS(df_train=df_model_train, 
                    df_test=df_model_test,
                    featuresCol='features',
                    labelCol='intrusion')

        accuracy_list = []
        class_1_recall_list = []
        class_0_recall_list = []
        models = []
        matrix_list = []
        report_list = []
        contamination_list = [0.01, 0.05, 0.1, 0.15, 0.17, 0.2, 0.33]

        # test models with different contamination levels of the training dataset, to return the model with the best results
        for contamination_level in contamination_list:
            model = hbos.train(contamination=contamination_level)
            accuracy, matrix, report = hbos.test(model)
            accuracy_list.append(accuracy)
            class_1_recall_list.append(report['1']['recall'])
            class_0_recall_list.append(report['0']['recall'])
            models.append(model)
            matrix_list.append(matrix)
            report_list.append(report)

            # NOTE: uncomment these lines to print the results for each contamination-case model
            #print(f'Recall (class 1) for contamination {contamination_level}: {round(report["1"]["recall"] * 100, 2)}%')
            #print(f'Recall (class 0) for contamination {contamination_level}: {round(report["0"]["recall"] * 100, 2)}%')
            #print(f"CM for contamination {contamination_level}:\n{matrix}")
            #print(f"Accuracy for contamination {contamination_level}: {round(accuracy * 100, 2)}%\n")
            #print(f"Report for contamination {contamination_level} and {n_bins} bins:\n{json.dumps(report, indent=4)}\n\n\n")

        # return the highest recall for class 1 (to ensure that the model detects as much anomalies as possible)
        max_class_1_recall = max(class_1_recall_list)
        indices_max_recall_1  = [i for i, r in enumerate(class_1_recall_list) if r == max_class_1_recall]
        best_index = indices_max_recall_1[0]
        min_recall_0 = class_0_recall_list[best_index]

        # if two models or more have the same highest recall for class 1, choose the model with the lowest recall for class 0
        # to ensure that we detect as much anomalies as possible. It's better to have some false positives than false negatives
        # We want to minimize false negatives (anomalies detected as normal packets) as much as possible
        # If the recall for class 0 is still the same between those models, choosing one model or another is the same thing,
        # since the accuracy is the same because the test dataset is the same
        for i in indices_max_recall_1[1:]:
            if class_0_recall_list[i] < min_recall_0:
                best_index = i
                min_recall_0 = class_0_recall_list[i]

        #recall_class_1 = class_1_recall_list[best_index]
        #recall_class_0 = class_0_recall_list[best_index]
        print(f"Contamination: {contamination_list[best_index] * 100}%")
        accuracy = accuracy_list[best_index]
        model = models[best_index]
        report = report_list[best_index]"""

        """### AUTOENCODER

        ae = Autoencoder(df_train=df_model_train, df_test=df_model_test)

        model = ae.train()

        accuracy, matrix, report = ae.test()"""

        """### ISOLATION FOREST
        
        if_class = IsolationForestLinkedIn(spark_session=self.__spark_session,
                                           df_train=df_model_train, 
                                           df_test=df_model_test, 
                                           featuresCol="features",
                                           scoreCol="score",
                                           predictionCol="predictionCol",
                                           labelCol="intrusion")

        model = if_class.train()

        accuracy, matrix, recall_class_1, df_model_test = if_class.test(model)"""

        ### One-Class SVM

        ocsvm = OneClassSVM(spark_session=self.__spark_session,
                            df_train=df_model_train,
                            df_test=df_model_test,
                            featuresCol="features",
                            predictionCol="prediction",
                            labelCol="intrusion")
        
        model = ocsvm.train()

        accuracy, evaluation, df_model_test = ocsvm.test(model)

        if evaluation is not None:
            print("Evaluation Report:\n")
            for key, value in evaluation.items():
                print(f"{key}: {value}")
        
        if accuracy is not None:
            print(f'Accuracy for model of device {dev_addr} for {dataset_type.value["name"].upper()}: {round((accuracy * 100), 2)}%')
        
        """if matrix is not None:
            print("Confusion matrix:\n", matrix)

        if recall_class_1 is not None:
            print(f"Recall (class 1): {round(recall_class_1 * 100, 2)}%", )
        
        if report is not None:
            print("Report:\n", json.dumps(report, indent=4)) # for sklearn models
            #print("Report:\n", report)"""

        # TODO uncomment after finishing all results' tables and after
        # fixing model replacement on sklearn, pyod and pytorch models!
        #self.__store_model(dev_addr, df_model_train, model, accuracy, dataset_type)

        dataset_train_path = f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_{dev_addr}_train.{datasets_format}'
        dataset_test_path = f'./generatedDatasets/{dataset_type.value["name"]}/lorawan_dataset_{dev_addr}_test.{datasets_format}'

        # Save final dataframe in JSON or PARQUET format (OPTIONAL)
        if datasets_format == "json":
            df_model_train.coalesce(1).write.mode("overwrite").json(dataset_train_path)
            df_model_test.coalesce(1).write.mode("overwrite").json(dataset_test_path)
        else:
            df_model_train.coalesce(1).write.mode("overwrite").parquet(dataset_train_path)
            df_model_test.coalesce(1).write.mode("overwrite").parquet(dataset_test_path)

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
                df_model_train, df_model_test = prepare_df_for_device(
                    self.__spark_session, dataset_type, dev_addr
                )  

                # If there are samples for the device, the model will be created
                if (df_model_train, df_model_test) != (None, None):
                    
                    # Processing phase
                    self.__create_model(df_model_train, df_model_test, dev_addr, dataset_type, datasets_format)         
        
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

