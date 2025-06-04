from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql.functions import col, avg, when, lit, monotonically_increasing_id
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from functools import reduce


# TODO: fix this, make this an anomaly / outlier detector rather than a traditional binary classifier


class KNNAnomalyDetector:
    def __init__(self, k, df_train, df_test, featuresCol, labelCol, predictionCol, threshold_percentile):
        self.__k = k
        self.__df_train = df_train.withColumn("id", monotonically_increasing_id())
        self.__df_test = df_test.withColumn("id", monotonically_increasing_id())
        self.__featuresCol = featuresCol
        self.__labelCol = labelCol
        self.__predictionCol = predictionCol
        self.__threshold_percentile = threshold_percentile
        self.__model_class = BucketedRandomProjectionLSH(
                                inputCol=featuresCol,
                                outputCol="hashes",
                                bucketLength=2.0,  # it can be changed
                                numHashTables=3
                            )

        self.__threshold = None


    def __get_neighbors_for_all(self, model, df_query, df_reference):
        spark = df_query.sparkSession
        neighbors_list = []

        for row in df_query.collect():
            point_id = row["id"]
            features = row[self.__featuresCol]

            # Create single-row DataFrame for the query point
            neighbors = model.approxNearestNeighbors(df_reference, features, self.__k + 1)
            neighbors = neighbors.filter(col("id") != lit(point_id))
            neighbors = neighbors.withColumn("query_id", lit(point_id))
            neighbors_list.append(neighbors)

        if neighbors_list:
            return reduce(lambda df1, df2: df1.union(df2), neighbors_list)
        else:
            return spark.createDataFrame([], df_reference.schema)

    def train(self):

        self.__model = self.__model_class.fit(self.__df_train)
        
        # Compute neighbors of training data with itself (excluding self-match)
        neighbors = self.__get_neighbors_for_all(self.__model, self.__df_train, self.__df_train)

        # Calculate average distances
        avg_dists = neighbors.groupBy("query_id").agg(avg("distCol").alias("avg_dist"))

        # Compute threshold based on percentile
        dist_percentiles = avg_dists.approxQuantile("avg_dist", [self.__threshold_percentile / 100], 0.01)
        self.__threshold = dist_percentiles[0]

        return self.__model

    def predict(self, model):

        # Compute neighbors of test points with training set
        neighbors = self.__get_neighbors_for_all(model, self.__df_test, self.__df_train)

        # Calculate average distance of each test point
        avg_dists = neighbors.groupBy("query_id").agg(avg("distCol").alias("avg_dist"))

        # Classify as anomaly based on threshold
        predictions = avg_dists.withColumn(
            self.__predictionCol,
            when(col("avg_dist") > self.__threshold, lit(1)).otherwise(lit(0))
        ).withColumnRenamed("query_id", "id")

        return predictions

    def test(self, model):
        
        predictions = self.predict(model)

        # Join with true labels
        labeled = self.__df_test.select("id", self.__labelCol).join(predictions, on="id", how="inner")

        y_true = [row[self.__labelCol] for row in labeled.collect()]
        y_pred = [row[self.__predictionCol] for row in labeled.collect()]

        report = classification_report(y_true, y_pred)
        matrix = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        return accuracy, matrix, report


"""################# draft ############
class KNNAnomalyDetection:

    ""
    It initializes the KNN model.
    
    Parameters:
        k (int): The number of neighbors to consider.
        train_data (pyspark dataframe): The training data.
        test_data (pyspark dataframe): The test data.

    ""
    def __init__(self, k, train_df, df_test, featuresCol, labelCol):
        self.__k = k
        self.__train_data = train_df.toPandas()
        self.__test_data = self.__df_test.toPandas()
        self.__featuresCol = featuresCol
        self.__labelCol = labelCol

    ""
    Predicts the label for a single observation.
    
    ""
    def predict(self, observation):

        distances = []
        obs_features = observation[self.__featuresCol]

        for _, train_obs in self.__train_data.iterrows():
            train_features = train_obs[self.__featuresCol]
            distance = sum((a - b) ** 2 for a, b in zip(obs_features, train_features)) ** 0.5
            distances.append((distance, train_obs[self.__labelCol]))

        distances.sort(key=lambda x: x[0])
        nearest_labels = [label for _, label in distances[:self.__k]]
        prediction = max(set(nearest_labels), key=nearest_labels.count)

        return prediction

    ""
    Evaluates the model on a labeled test dataset.
    Returns accuracy and total samples.

    ""
    def test(self):

        correct = 0
        true_labels = []
        predictions = []

        for _, test_obs in self.__test_data.iterrows():
            pred = self.predict(test_obs)
            true_label = test_obs[self.__labelCol]

            if pred == true_label:
                correct += 1

            predictions.append(pred)
            true_labels.append(true_label)

        total = len(self.__test_data)
        accuracy = correct / total if total > 0 else 0.0

        labels = sorted(list(set(true_labels + predictions)))  # todas as classes possíveis
        matrix = confusion_matrix(true_labels, predictions, labels=labels)
        report = classification_report(true_labels, predictions, labels=labels)

        return accuracy, matrix, labels, report"""