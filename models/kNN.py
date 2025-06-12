from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql.functions import col, avg, when, lit, monotonically_increasing_id
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from functools import reduce
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number


# TODO: fix this, make this an anomaly / outlier detector rather than a traditional binary classifier


class KNNAnomalyDetector:
    def __init__(self, k, df_train, df_test, featuresCol, labelCol, predictionCol, threshold_percentile=95):
        self.__k = k
        self.__df_train = df_train.withColumn("id", monotonically_increasing_id()).select(
            "id", featuresCol, labelCol, predictionCol
        )
        self.__df_test = df_test.withColumn("id", monotonically_increasing_id()).select(
            "id", featuresCol, labelCol, predictionCol
        )
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

    
    def __compute_avg_distances(self, model, df_query, df_reference, is_train=False):

        # Join aproximado entre query e referência
        joined = model.approxSimilarityJoin(
            df_query.select("id", self.__featuresCol),
            df_reference.select("id", self.__featuresCol),
            float("inf"),  # pegar todas as distâncias
            distCol="distCol"
        )
        # Remove self-match if it's training data
        if is_train:
            joined = joined.filter(col("datasetA.id") != col("datasetB.id"))

        windowSpec = Window.partitionBy("datasetA.id").orderBy("distCol")
        neighbors_ranked = joined.withColumn("rank", row_number().over(windowSpec))

        # Pega apenas os k vizinhos mais próximos
        top_k = neighbors_ranked.filter(col("rank") <= self.__k)

        # Média das distâncias para cada observação da query
        avg_dists = top_k.groupBy("datasetA.id").agg(avg("distCol").alias("avg_dist"))

        return avg_dists.withColumnRenamed("datasetA.id", "query_id")


    def train(self):

        self.__model = self.__model_class.fit(self.__df_train)
        
        # Calculate average distances
        avg_dists = self.__compute_avg_distances(self.__model, self.__df_train, self.__df_train, is_train=True)

        # Compute threshold based on percentile
        dist_percentiles = avg_dists.approxQuantile("avg_dist", [self.__threshold_percentile / 100], 0.01)
        self.__threshold = dist_percentiles[0]

        return self.__model

    def predict(self, model=None):

        model = model or self.__model
        avg_dists = self.__compute_avg_distances(model, self.__df_test, self.__df_train)

        # Classify based on threshold
        predictions = avg_dists.withColumn(
            self.__predictionCol,
            when(col("avg_dist") > self.__threshold, lit(1)).otherwise(lit(0))
        ).withColumnRenamed("query_id", "id")

        return predictions

    def test(self, model):

        if self.__df_test is None:
            return None, None, None
        
        predictions = self.predict(model)

        # Join with true labels
        labeled = self.__df_test.select("id", self.__labelCol).join(predictions, on="id", how="inner")

        y_true = [row[self.__labelCol] for row in labeled.collect()]
        y_pred = [row[self.__predictionCol] for row in labeled.collect()]

        report = classification_report(y_true, y_pred, zero_division=0)
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