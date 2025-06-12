
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, IntegerType




class IsolationForest:

    def __init__(self, spark_session, df_train, df_test, featuresCol, scoreCol, 
                 predictionCol, labelCol, intrusion_rate, numTrees=2000,
                 maxFeatures=1.0, seed=42):
        
        self.__spark_session = spark_session
        self.__df_train = df_train.drop("prediction", "score")
        self.__df_test = df_test.drop("prediction", "score")
        self.__predictionCol = predictionCol
        self.__featuresCol = featuresCol
        self.__scoreCol = scoreCol
        self.__labelCol = labelCol
        self.__intrusionRate = intrusion_rate

        # Build Java IsolationForest Estimator
        self.__model = spark_session._jvm.com.linkedin.relevance.isolationforest.IsolationForest() \
                .setNumEstimators(numTrees) \
                .setMaxSamples(float(min(256, df_train.count()))) \
                .setBootstrap(False) \
                .setMaxFeatures(float(maxFeatures)) \
                .setFeaturesCol(featuresCol) \
                .setPredictionCol(predictionCol) \
                .setScoreCol(scoreCol) \
                .setContamination(0.0) \
                .setContaminationError(0.0) \
                .setRandomSeed(seed)


    """
    Fits the Isolation Forest model using training data.
    
    """
    def train(self):
        return self.__model.fit(self.__df_train.select(self.__featuresCol, self.__labelCol)._jdf)

    """
    Apply the fitted model to a new dataset (e.g., test set).
    
    """
    def predict(self, model, df):
        if model is None:
            raise RuntimeError("Model does not exist!")
        
        java_df = model.transform(df)
        return DataFrame(java_df, self.__spark_session)
    
    def evaluate(self, df_with_preds):

        # Define um limiar com base nos scores (ex: top 5% mais anómalos)
        threshold = df_with_preds.approxQuantile(self.__scoreCol, [self.__intrusionRate], 0.01)[0]

        self.__df_test = df_with_preds \
            .withColumn(self.__predictionCol, (col(self.__scoreCol) < threshold).cast(DoubleType()))

        evaluator = MulticlassClassificationEvaluator(
            labelCol=self.__labelCol,
            predictionCol=self.__predictionCol,
            metricName="accuracy"
        )

        accuracy = evaluator.evaluate(self.__df_test)

        self.__df_test = self.__df_test \
            .withColumn(self.__predictionCol, col(self.__predictionCol).cast(IntegerType()))

        # Confusion Matrix
        cm = self.__df_test.groupBy(self.__labelCol, self.__predictionCol).count().collect()
        matrix = {(row[self.__labelCol], row[self.__predictionCol]): row["count"] for row in cm}
        
        tn = matrix.get((0, 0), 0)
        fp = matrix.get((0, 1), 0)
        fn = matrix.get((1, 0), 0)
        tp = matrix.get((1, 1), 0)

        confusion_matrix = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

        return accuracy, confusion_matrix, self.__df_test
    
    def test(self, model):

        if self.__df_test is None:
            print("Test dataset is empty. Skipping testing.")
            return None, None

        df_predictions = self.predict(model, self.__df_test._jdf)

        return self.evaluate(df_predictions)