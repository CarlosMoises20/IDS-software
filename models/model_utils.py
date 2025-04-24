
import mlflow
from mlflow.tracking import MlflowClient
from models.autoencoder import Autoencoder
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow.pyspark.ml as mlflow_pyspark
import mlflow.sklearn as mlflow_sklearn
from mlflow.models import infer_signature
import mlflow, shutil, os
from pyspark import SparkContext as sc

class ModelUtils:

    def __init__(self):
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
    def __sample_random_split(self, pdf_model, seed):

        total_count = len(pdf_model)

        # If there is only one sample for the device, use that sample for training, 
        # and don't apply testing for that model
        if total_count == 1:
            pdf_train, pdf_test = pdf_model, None

        # If there are between 2 and 9 samples, split the samples for training and testing by 50-50
        if 2 <= total_count < 10:
            pdf_train = pdf_model.sample(frac=0.5, random_state=seed)

        # If there are between 10 and 20 samples, split the samples for training and testing by 70-30
        elif total_count < 20:
            pdf_train = pdf_model.sample(frac=0.7, random_state=seed)

        # If there are 20 or more samples, split the samples for training and testing by 85-15
        else:
            pdf_train = pdf_model.sample(frac=0.85, random_state=seed)

        pdf_test = pdf_model.drop(pdf_train.index)

        return pdf_train, pdf_test



    """
    This function creates a ML model based on a given DevAddr, and stores it on MLFlow
    It uses, as input, all samples of the pandas dataframe 'pdf' whose DevAddr is equal to 'dev_addr'

    """
    def create_model(self, pdf, dev_addr):

        ae = Autoencoder(pdf, dev_addr)

        ae.train()

        pdf = ae.label_data_by_reconstruction_error()

        # randomly divide dataset into training and test, according to the total number of examples 
        # and set a seed in order to ensure reproducibility, which is important to 
        # ensure that the model is always trained and tested on the same examples each time the
        # model is run. This is important to compare the model's performance in different situations
        pdf_train, pdf_test = self.__sample_random_split(pdf_model=pdf, seed=522)

        X_train = list(pdf_train["features_dense"])
        y_train = pdf_train["intrusion"]
        X_test = list(pdf_test["features_dense"])
        y_test = pdf_test["intrusion"]

        rf = RandomForestClassifier(n_estimators=30)
        rf_model = rf.fit(X_train, y_train)

        #rf = RandomForestClassifier(numTrees=30, featuresCol="features_dense", labelCol="intrusion")
        #rf_model = rf.fit(pdf_train)

        if pdf_test is not None:

            #results = rf_model.evaluate(pdf_test)
            #accuracy = results.accuracy
            accuracy = rf.score(X_test, y_test)


        
        """ LOGISTIC REGRESSION
        
        lr = LogisticRegression(featuresCol="features", labelCol="intrusion", 
                                regParam=0.1, elasticNetParam=1.0,
                                family="multinomial", maxIter=50)
        

        lr_model = lr.fit(pdf_train)

        results = lr_model.evaluate(pdf_test)
        
        """
        
        """if results is not None:
            print(f"accuracy: {accuracy:.2f}")
            print(f"precision for each label: {results.precisionByLabel}")"""
        
        if accuracy is not None:
            print(f"accuracy: {accuracy:.2f}")
        
        
        """  KMEANS

        # Apply clustering (KMeans or, as alternative, DBSCAN) to divide samples into clusters according to the density
        k_means = KMeans(k=3, seed=522, maxIter=100)

        # TODO: think if covering this with a try/except block wouldn't be better
        model = k_means.fit(pdf_train)

        predictions = model.transform(pdf_test)
        
        # Evaluate the model
        evaluator = ClusteringEvaluator()
        
        accuracy = evaluator.evaluate(predictions)

        accuracy_list.append(accuracy)

        """

        
        # Verify if a model associated to the device already exists. If so, return it;
        # otherwise, return None
        mlflow_retrieved_model, old_run_id = self.__get_model_by_devaddr(dev_addr)

        signature = infer_signature(pdf_test, pdf)

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
            #mlflow_pyspark.autolog()
            mlflow_sklearn.autolog()
            #mlflow.spark.log_model(rf_model, "model", signature=signature)
            mlflow.sklearn.log_model(rf_model, "model", signature=signature)


        print(f"Model for end-device with DevAddr {dev_addr} saved successfully")
