
import time
from processing.processing import DataProcessing
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Model
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from auxiliaryFunctions.general import get_all_attributes_names, format_time
from crate.client import connect
from constants import *



# TODO: this code seems to be generic for all dataset types; therefore,
# we will probably remove DataProcessing, RxpkDataProcessing and TxpkDataProcessing classes, and pass
# this code to MessageClassification class or something 
class RxpkProcessing(DataProcessing):
    
    @staticmethod
    def process_data(df):

        start_time = time.time()

        # Cast DevAddr column to integer and get distinct values
        dev_addr_list = df.select("DevAddr").filter(df["DevAddr"].isNotNull()).distinct()

        # Convert to a list of integers
        dev_addr_list = [row.DevAddr for row in dev_addr_list.collect()]

        # get names of all attributes names to assemble since they are all now numeric (excluding DevAddr)
        column_names = list(set(get_all_attributes_names(df.schema)) - set(["DevAddr"]))

        # list of all models' accuracy
        accuracy_list = []

        ### Initialize connection with CrateDB
        db_connection = connect(CRATEDB_URI)
        cursor = db_connection.cursor()


        # Create a specific model for each device (DevAddr)
        for dev_addr in dev_addr_list:

            # Filter dataset by the selected DevAddr
            df_model = df.filter(df.DevAddr == dev_addr)

            df_model = df_model.drop("DevAddr")

            # Create the VectorAssembler that merges all features of the dataset into a Vector
            # These feature are, now, all numeric and with the missing values all imputed, so now we can use them
            assembler = VectorAssembler(inputCols=column_names, outputCol="features")
    
            pipeline = Pipeline(stages=[assembler])

            # Train the pipeline model to assemble all features
            pipelineModel = pipeline.fit(df_model)

            # Apply the pipeline to the dataset to assemble the features
            df_model = pipelineModel.transform(df_model)

            # randomly divide dataset into training (85%) and test (15%)
            # and set a seed in order to ensure reproducibility, which is important to 
            # ensure that the model is always trained and tested on the same examples each time the
            # model is run. This is important to compare the model's performance in different situations
            # (this proportion can be modified according to the results)
            df_model_train, df_model_test = df_model.randomSplit([0.85, 0.15], seed=522)

            # TODO: review where to make this step
            # Filter dataset by the non-intrusion messages, which are the ones that will be used to train the model
            # This ensures that the model learns the network traffic patterns of normal behaviour, and then it is capable
            # to analyze new messages and distinguish between expected and suspicious activity
            df_model_train = df_model_train.filter(df_model_train.intrusion == 0)

            # TODO: review this approach
            # Aggregate test samples with train samples where intrusion=1, that were not used in training
            df_model_test = df_model_train.filter(df_model_train.intrusion == 1).union(df_model_test)

            # Create the Random Forest Classifier model
            rf = RandomForestClassifier(featuresCol="features", labelCol="intrusion",
                                        predictionCol="prediction", 
                                        numTrees=6, maxDepth=3, seed=522, maxMemoryInMB=1024)

            # Train the model using the training data
            model = rf.fit(df_model_train)

            # Fazer previsões
            predictions = model.transform(df_model_test)

            # Avaliar o modelo
            evaluator = BinaryClassificationEvaluator(labelCol="intrusion", metricName="areaUnderROC",
                                                      rawPredictionCol="prediction")
            
            roc_auc = evaluator.evaluate(predictions)

            accuracy_list.append(roc_auc)

            # Check if model of 'DevAddr' for 'RXPK' already exists
            cursor.execute("SELECT * FROM model WHERE dev_addr = ? AND dataset_type = 'RXPK'", [dev_addr])

            # If exists, update it
            if cursor.rowcount > 0:
                cursor.execute("UPDATE model SET model = ?, accuracy_list = ? WHERE dev_addr = ? AND dataset_type = 'RXPK'", [model, roc_auc, dev_addr])
            
            # If not, insert it
            else:
                # TODO: fix
                cursor.execute("INSERT INTO model(dev_addr, dataset_type, model, accuracy_list) VALUES (?, 'RXPK', ?, ?)", [dev_addr, model, roc_auc])


        end_time = time.time()


        print("Average accuracy_list of models:", (sum(accuracy_list) / len(accuracy_list)) * 100, "%")

        # Print the total time of pre-processing; the time is in seconds, minutes or hours
        print("Time of rxpk processing:", format_time(end_time - start_time), "\n\n")

        # TODO: the model and results should be returned to be used on real-time incoming messages
        return 3