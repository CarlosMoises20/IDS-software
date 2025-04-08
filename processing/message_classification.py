
from common.auxiliary_functions import bind_dir_files, get_all_attributes_names, format_time
from common.constants import CRATEDB_URI
from prepareData.prepareData import prepare_dataset
from crate.client import connect
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, asc, desc
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark import SparkContext
import time


class MessageClassification:

    def __init__(self, spark_session):
        self.__spark_session = spark_session


    def __train_test_model(self, df, dev_addr, accuracy_list, cursor):
        
        # Filter dataset considering the selected DevAddr
        # Remove DevAddr to make processing more efficient, since we don't need it anymore 
        df_model = df.filter(df.DevAddr == dev_addr).drop("DevAddr")

        # randomly divide dataset into training (80%) and test (20%)
        # and set a seed in order to ensure reproducibility, which is important to 
        # ensure that the model is always trained and tested on the same examples each time the
        # model is run. This is important to compare the model's performance in different situations
        df_model_train, df_model_test = df_model.randomSplit([0.8, 0.2], seed=522)

        # Apply clustering (KMeans or, as alternative, DBSCAN) to divide samples into clusters according to the density
        k_means = KMeans(k=2, seed=522, maxIter=100)

        model = k_means.fit(df_model_train)

        predictions = model.transform(df_model_test)

        print(predictions)

        # TODO: fix and complete 
        
        # Evaluate the model
        evaluator = BinaryClassificationEvaluator(labelCol="intrusion", metricName="areaUnderROC",
                                                    rawPredictionCol="prediction")
        
        accuracy = evaluator.evaluate(predictions)

        accuracy_list.append(accuracy)

        # Check if model of 'DevAddr' for the specified dataset type already exists
        cursor.execute("SELECT * FROM model WHERE dev_addr = ?", [dev_addr])

        # If exists, update it
        if cursor.rowcount > 0:
            cursor.execute("UPDATE model SET model = ?, accuracy = ? WHERE dev_addr = ?", 
                            [model, accuracy, dev_addr])
        
        # If not, insert it
        else:
            # TODO: fix
            cursor.execute("INSERT INTO model(dev_addr, model, accuracy) VALUES (?, ?, ?)", 
                            [dev_addr, model, accuracy])

        


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
        
        # Initialize SparkContext for models' parallel processing
        sc = SparkContext.getOrCreate()

        ### Begin processing
        start_time = time.time()

        # Cast DevAddr column to integer and get distinct values
        dev_addr_list = df.select("DevAddr").filter(df["DevAddr"].isNotNull()).distinct()

        # Convert to a list of integers
        dev_addr_list = [row.DevAddr for row in dev_addr_list.collect()]

        # list of all models' accuracy to be used to return the mean accuracy of all models
        accuracy_list = []

        ### Initialize connection with CrateDB
        db_connection = connect(CRATEDB_URI)

        cursor = db_connection.cursor()

        for dev_addr in dev_addr_list:
            self.__train_test_model(df, dev_addr, accuracy_list, cursor)

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

        
