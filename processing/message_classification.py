
from common.auxiliary_functions import bind_dir_files, get_all_attributes_names, format_time
from common.constants import CRATEDB_URI
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

    @staticmethod
    def train_test_model(df, dev_addr, column_names, cursor, dataset_type, accuracy_list):
        
        # Filter dataset considering the selected DevAddr
        # Remove DevAddr to make processing more efficient, since we don't need it anymore 
        df_model = df.filter(df.DevAddr == dev_addr).drop("DevAddr")

        # Create the VectorAssembler that merges all features of the dataset into a Vector
        # These feature are, now, all numeric and with the missing values all imputed, so now we can use them
        assembler = VectorAssembler(inputCols=column_names, outputCol="features")

        pipeline = Pipeline(stages=[assembler])

        # Train and apply the pipeline model to assemble all features
        df_model = pipeline.fit(df_model).transform(df_model)

        # randomly divide dataset into training (85%) and test (15%)
        # and set a seed in order to ensure reproducibility, which is important to 
        # ensure that the model is always trained and tested on the same examples each time the
        # model is run. This is important to compare the model's performance in different situations
        df_model_train, df_model_test = df_model.randomSplit([0.85, 0.15], seed=522)

        # Apply clustering (KMeans or, as alternative, DBSCAN) to divide samples into clusters according to the density
        k_means = KMeans(k=2, seed=522, maxIter=100)

        k_means_model = k_means.fit(df_model_train)

        k_means_predictions = k_means_model.transform(df_model_test)

        print(k_means_predictions)

        k_means_predictions.select("tmst", "DevAddr", "FCnt").sort(asc("tmst")).show(truncate=False, vertical=True)

        # Apply Autoencoder to learn to reconstruct the original data
        #ae_model = Autoencoder(df_model_train, df_model_test)

        #ae_model.train(num_epochs=20, learning_rate=0.7, momentum=0.0005)


        """
        # TODO: fix and complete 

        
        # Evaluate the model
        evaluator = BinaryClassificationEvaluator(labelCol="intrusion", metricName="areaUnderROC",
                                                    rawPredictionCol="prediction")
        
        accuracy = evaluator.evaluate(predictions)

        accuracy_list.append(accuracy)

        # Check if model of 'DevAddr' for the specified dataset type already exists
        cursor.execute("SELECT * FROM model WHERE dev_addr = ? AND dataset_type = ?", [dev_addr, dataset_type.value["name"]])

        # If exists, update it
        if cursor.rowcount > 0:
            cursor.execute("UPDATE model SET model = ?, accuracy = ? WHERE dev_addr = ? AND dataset_type = ?", 
                            [model, accuracy, dev_addr, dataset_type.value["name"]])
        
        # If not, insert it
        else:
            # TODO: fix
            cursor.execute("INSERT INTO model(dev_addr, dataset_type, model, accuracy) VALUES (?, ?, ?, ?)", 
                            [dev_addr, dataset_type.value["name"], model, accuracy])
            

        """



    """
    Function to execute the IDS

    It receives the spark session (spark_session) that handles the dataset processing and
    the corresponding dataset type (dataset_type) defined by DatasetType Enum

    It returns the processing results, namely the accuracy and the confusion matrix that show the
    model performance

    """
    @staticmethod
    def message_classification(spark_session, dataset_type):

        # Define the dataset root path
        dataset_root_path = "./datasets"
        
        ### Bind all log files into a single log file if it doesn't exist yet,
        ### to simplify data processing
        combined_logs_filename = bind_dir_files(dataset_root_path, dataset_type)

        # Load the dataset into a Spark Dataframe
        df = spark_session.read.json(combined_logs_filename)

        # Initialize pre-processing class
        pre_processing_class = dataset_type.value["pre_processing_class"]

        ### Begin pre-processing based on the dataset type
        df = pre_processing_class.pre_process_data(df)


        # Initialize SparkContext for parallel processing
        sc = SparkContext.getOrCreate()

        ### Begin processing (general for all dataset types)

        start_time = time.time()

        # Cast DevAddr column to integer and get distinct values
        dev_addr_list = df.select("DevAddr").filter(df["DevAddr"].isNotNull()).distinct()

        # Convert to a list of integers
        dev_addr_list = [row.DevAddr for row in dev_addr_list.collect()]

        # get, in a list of strings, the names of all attributes names to assemble (excluding DevAddr)
        # since they are all now numeric
        column_names = list(set(get_all_attributes_names(df.schema)) - set(["DevAddr"]))

        # list of all models' accuracy to be used to return the mean accuracy of all models
        accuracy_list = []

        ### Initialize connection with CrateDB
        db_connection = connect(CRATEDB_URI)

        cursor = db_connection.cursor()

        # Use SparkContext for parallel processing of all models
        dev_addr_rdd = sc.parallelize(dev_addr_list)

        # Create a specific model for each device (DevAddr), paralellizing the process to make it faster,
        # since these are processes are independent from each other
        dev_addr_rdd.map(
            lambda dev_addr: MessageClassification.train_test_model(
                    df, dev_addr, column_names, cursor, dataset_type, accuracy_list
                )
        ).collect()

        # Close connection to CrateDB
        cursor.close()
        db_connection.close()

        end_time = time.time()

        avg_accuracy = (sum(accuracy_list) / len(accuracy_list)) * 100

        print("Average accuracy_list of", dataset_type.value["name"], "models:", avg_accuracy, "%")

        # Print the total time of pre-processing; the time is in seconds, minutes or hours
        print("Time of processing in", dataset_type.value["name"], ":", format_time(end_time - start_time), "\n\n")

        # TODO: change return for a general confusion matrix??
        return avg_accuracy

        
