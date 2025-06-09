
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, IntegerType


## it's finally working reasonably!!! TODO: try to make improvements on code organization, if possible

class IsolationForest:

    def __init__(self, spark_session, df_train, df_test, featuresCol, scoreCol, 
                 predictionCol, labelCol, dataset_type, dev_addr,
                 numTrees=700, maxSamples=1.0, maxFeatures=1.0, contamination=0.1, contaminationError=0.001):
        
        self.__spark_session = spark_session
        self.__df_train = df_train.drop("prediction", "score")
        self.__df_test = df_test.drop("prediction", "score")
        self.__predictionCol = predictionCol
        self.__labelCol = labelCol
        self.__dataset_type = dataset_type
        self.__dev_addr = dev_addr

        # Build Java IsolationForest Estimator
        self.__model = spark_session._jvm.com.linkedin.relevance.isolationforest.IsolationForest() \
                .setNumEstimators(numTrees) \
                .setMaxSamples(float(maxSamples)) \
                .setMaxFeatures(float(maxFeatures)) \
                .setFeaturesCol(featuresCol) \
                .setPredictionCol(predictionCol) \
                .setScoreCol(scoreCol) \
                .setContamination(contamination) \
                .setContaminationError(contaminationError)
        
    """
    Fits the Isolation Forest model using training data.
    
    """
    def train(self):
        return self.__model.fit(self.__df_train._jdf)

    """
    Apply the fitted model to a new dataset (e.g., test set).
    
    """
    def predict(self, model, df):
        if model is None:
            raise RuntimeError("Model does not exist!")
        
        java_df = model.transform(df)
        return DataFrame(java_df, self.__spark_session)
    
    def evaluate(self, df_with_preds):

        df_eval = df_with_preds \
            .withColumn(self.__predictionCol, col(self.__predictionCol).cast(DoubleType()))

        evaluator = MulticlassClassificationEvaluator(
            labelCol=self.__labelCol,
            predictionCol=self.__predictionCol,
            metricName="accuracy"
        )

        accuracy = evaluator.evaluate(df_eval)

        df_eval.select("tmst", self.__predictionCol, self.__labelCol) \
                .withColumn(self.__predictionCol, col(self.__predictionCol).cast(IntegerType())) \
                .write.mode("overwrite").json(f"./generatedDatasets/{self.__dataset_type}/results_{self.__dev_addr}.json")

        # Confusion Matrix
        cm = df_eval.groupBy(self.__labelCol, self.__predictionCol).count().collect()
        matrix = {(row[self.__labelCol], row[self.__predictionCol]): row["count"] for row in cm}
        
        tn = matrix.get((0, 0), 0)
        fp = matrix.get((0, 1), 0)
        fn = matrix.get((1, 0), 0)
        tp = matrix.get((1, 1), 0)

        confusion_matrix = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

        return accuracy, confusion_matrix
    
    def test(self, model):

        df_predictions = self.predict(model, self.__df_test._jdf)

        return self.evaluate(df_predictions)