
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql.functions import col, avg, when, lit, monotonically_increasing_id
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number


class SparkKNN:
    
    def __init__(self, spark_session, k, df_train, df_test, featuresCol, labelCol, predictionCol):

        self.__spark_session = spark_session
        self.__df_train = df_train.select(featuresCol, labelCol)
        self.__df_test = df_test.select(featuresCol, labelCol)
        self.__labelCol = labelCol
        self.__predictionCol = predictionCol
        self.__class_instance = spark_session._jvm.org.apache.spark.ml.classification.kNN_IS.__getattr__("kNN_ISClassifier$")
        self.__model_class = self.__class_instance \
                                    .setK(k) \
                                    .setFeaturesCol(featuresCol) \
                                    .setLabelCol(labelCol) \
                                    .setPredictionCol(predictionCol)


    def train(self):
        return self.__model_class.train(self.__df_train)
    
    def test(self, model):
        df_preds = self.predict(model)
        return self.evaluate(df_preds)

    def predict(self, model):

        if model is None:
            raise Exception("Model must be created first!")

        preds_jdf = model.transform(self.__df_test._jdf)

        return self.__spark_session.createDataFrame(preds_jdf)
    

    def evaluate(self, df_preds):

        # Join with true labels
        df_true = self.__df_test.withColumn("id", monotonically_increasing_id())
        df_pred = df_preds.withColumn("id", monotonically_increasing_id())

        joined = df_true.select(self.__labelCol, "id") \
                        .join(df_pred.select(self.__predictionCol, "id"), on="id")
        
        rows = joined.select(self.__labelCol, self.__predictionCol).collect()
        y_true = [row[self.__labelCol] for row in rows]
        y_pred = [row[self.__predictionCol] for row in rows]

        report = classification_report(y_true, y_pred, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        return accuracy, matrix, report


"""
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
                                bucketLength=1.5,  # it can be changed
                                numHashTables=3
                            )

        self.__threshold = None        

    def __compute_avg_distances(self, model, df_query, df_reference, is_train=False):

        # Approximated join between query and reference
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

        model = self.__model_class.fit(self.__df_train)
        
        # Calculate average distances
        avg_dists = self.__compute_avg_distances(model, self.__df_train, self.__df_train, is_train=True)

        # Compute threshold based on percentile
        dist_percentiles = avg_dists.approxQuantile("avg_dist", [self.__threshold_percentile / 100], 0.01)
        self.__threshold = dist_percentiles[0]

        return model

    def predict(self, model):

        if model is None:
            raise Exception("Model must be created first!")

        avg_dists = self.__compute_avg_distances(model, self.__df_test, self.__df_train)

        # Classify based on threshold
        predictions = avg_dists.withColumn(
            self.__predictionCol,
            when(col("avg_dist") > self.__threshold, lit(1)).otherwise(lit(0))
        ).withColumnRenamed("query_id", "id")

        return predictions
    
    def evaluate(self, df_preds):

        # Join with true labels
        labeled = self.__df_test.select("id", self.__labelCol).join(df_preds, on="id", how="inner")

        y_true = [row[self.__labelCol] for row in labeled.collect()]
        y_pred = [row[self.__predictionCol] for row in labeled.collect()]

        report = classification_report(y_true, y_pred, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        return accuracy, matrix, report

    def test(self, model):

        if self.__df_test is None:
            return None, None, None
        
        predictions = self.predict(model)

        return self.evaluate(predictions)"""