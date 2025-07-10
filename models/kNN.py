
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql.functions import col, avg, when, lit, monotonically_increasing_id
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

"""
This class represents a kNN anomaly-based implementation based on PySpark library BucketedRandomProjectionLSH, a class for
Euclidean distance metrics. It calculates the distances from one point to the k nearest points to determine if the point
corresponds to an anomaly or not. This implementation is only feasible for relatively small datasets, since it calculates the distance
of each sample to its nearest neighbors, which is extremely inefficient for large datasets.

"""
class KNNAnomalyDetector:

    """
    This class is initialized with the following parameters:

        df_train: the Spark dataframe corresponding to the train dataset
        
        df_test: the Spark dataframe corresponding to the test dataset
        
        featuresCol: the name of the column in the Spark dataframes of train and test datasets that contains all the features of each
                    example of the dataset that are used by the model for training and testing

        labelCol: the name of the column in train and test datasets that corresponds to the label of the dataset. Since this is an
                    unsupervised algorithm, this "label" is only used in testing to compute the evaluation metrics that measure the efficacy
                    of the model during testing 

        predictionCol: the name of the column in train and test datasets that will contain the predictions of the model during testing

        threshold_percentile (default=0.99): threshold percentile that will be used to compute the adequate threshold for anomaly detection

        threshold (default=NOne): threshold that will be calculated based on the percentile 'threshold_percentile'

        model_class: variable with the Spark class corresponding to the model used for training
    
    """
    def __init__(self, df_train, df_test, featuresCol, labelCol, predictionCol, threshold_percentile=0.9):
        self.__k = max(5, min(round(df_train.count() * 0.1), 15))
        
        # NOTE uncomment if you want to print the value of 'k' 
        #print("k:", self.__k)
        
        self.__df_train = df_train
        self.__df_test = df_test
        self.__featuresCol = featuresCol
        self.__labelCol = labelCol
        self.__predictionCol = predictionCol
        self.__threshold_percentile = threshold_percentile

        # bucketLength: the length of each interval in the random projections; it defines the granularity of the division
        # of the vetorial space
        # numHashTables: number of independent hash tables to be used; more hash tables increase the probability to find
        # more near neighbors, since two points will more likely be in the same bucket in at least one of the projections
        self.__model_class = BucketedRandomProjectionLSH(
                                inputCol=featuresCol,
                                outputCol="hashes",
                                bucketLength=1.0,
                                numHashTables=5  
                            )

        self.__threshold = None        

    """
    This method computes the average distances using the model on the parameter, a dataframe used for query
    and a dataframe used for reference; the distances are calculated between the points of 'df_query' and the
    points of 'df_reference'
    
    """
    def __compute_avg_distances(self, model, df_query, df_reference, is_train=False):

        # Approximated join between query and reference
        joined = model.approxSimilarityJoin(
            df_query.select("id", self.__featuresCol),
            df_reference.select("id", self.__featuresCol),
            float("inf"),  # take every distances
            distCol="distCol"
        )

        # Remove self-match if it's training data
        if is_train:
            joined = joined.filter(col("datasetA.id") != col("datasetB.id"))

        windowSpec = Window.partitionBy("datasetA.id").orderBy("distCol")
        neighbors_ranked = joined.withColumn("rank", row_number().over(windowSpec))

        # Takes only the k nearest neighbors
        top_k = neighbors_ranked.filter(col("rank") <= self.__k)

        # Distance averages for each query observation
        avg_dists = top_k.select(col("datasetA.id").alias("query_id"), col("distCol")) \
                        .groupBy("query_id") \
                        .agg(avg("distCol").alias("avg_dist"))

        all_queries = df_query.select(col("id").alias("query_id"))

        avg_dists = all_queries.join(avg_dists, on="query_id", how="left")

        return avg_dists

    """"
    Fits the kNN model using training data.
    
    """
    def train(self):

        # Add "id" column on the dataset do uniquely identify each sample
        self.__df_train = self.__df_train.withColumn("id", monotonically_increasing_id()).select(
            "id", self.__featuresCol, self.__labelCol, self.__predictionCol
        )

        model = self.__model_class.fit(self.__df_train)
        
        # Calculate average distances
        avg_dists = self.__compute_avg_distances(model, self.__df_train, self.__df_train, is_train=True)

        # Compute threshold based on percentile
        dist_percentiles = avg_dists.approxQuantile("avg_dist", [self.__threshold_percentile], 0.001)
        self.__threshold = dist_percentiles[0] if dist_percentiles else 0.0
        
        # NOTE: uncomment this line to print the computed threshold
        #print("threshold:", self.__threshold)

        return model

    """
    Apply the fitted model to a new dataset (e.g., test set).
    
    """
    def predict(self, model):

        # If a model is not passed, an exception is thrown
        if model is None:
            raise Exception("Model must be created first!")
        
        # Add "id" column on the dataset do uniquely identify each sample
        self.__df_test = self.__df_test.withColumn("id", monotonically_increasing_id() + self.__df_train.count()).select(
            "id", self.__featuresCol, self.__labelCol, self.__predictionCol
        )

        # Calculate the distances between each point in the test dataset and the k nearest neighbors in training dataset
        avg_dists = self.__compute_avg_distances(model, self.__df_test, self.__df_train)

        # Classify based on threshold
        predictions = avg_dists.withColumn(
            self.__predictionCol,
            when(col("avg_dist") > self.__threshold, lit(1)).otherwise(lit(0))
        ).withColumnRenamed("query_id", "id")

        return predictions
    
    """
    This method evaluates the predictions calculated by the model during testing, to give an idea of the model's efficacy
    As an argument, the method receives 'df_preds', which corresponds to a Spark dataframe that contains the model's calculated predictions
    It returns the accuracy, a dictionary that contains the confusion matrix, i.e., true negatives (tn), true positives (tp), false positives (fp) and false negatives (fn),
    and the report that contains the most relevant evaluation metrics

    """
    def evaluate(self, df_preds):

        # Join with true labels
        labeled = self.__df_test.select("id", self.__labelCol).join(df_preds, on="id", how="inner")

        y_true = [row[self.__labelCol] for row in labeled.collect()]        # Real / Expected labels
        y_pred = [row[self.__predictionCol] for row in labeled.collect()]   # Predicted labels
        
        # Report, with relevant evaluation metrics such as F1-Score, Recall and Precision
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        matrix = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

        # Accuracy, to determine the level of convergence between the predicted labels and the expected labels
        accuracy = accuracy_score(y_true, y_pred)

        return accuracy, matrix, report

    """
    This method corresponds to the test of the model. This method will call the predict method
    to calculate the predictions using the trained model, and then it will call the evaluate method to compute and return the evaluation metrics

    """
    def test(self, model):        
        predictions = self.predict(model)
        return self.evaluate(predictions)