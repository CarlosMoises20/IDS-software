
from pyspark.ml.wrapper import JavaTransformer
from pyspark.sql.functions import col, when
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


class IsolationForest:

    def __init__(self, spark_session, df_train, df_test, featuresCol, predictionCol, labelCol, 
                 contamination=0.05, numTrees=100, maxSamples=256, seed=42):
        self.__df_train = df_train
        self.__df_test = df_test
        self.__featuresCol = featuresCol
        self.__predictionCol = predictionCol
        self.__labelCol = labelCol
        self.__model = JavaTransformer._create_from_java_class(
                                spark_session._jvm.com.linkedin.relevance.IsolationForest()
                                    .setNumTrees(numTrees)
                                    .setMaxSamples(maxSamples)
                                    .setFeaturesCol(featuresCol)
                                    .setPredictionCol(predictionCol)
                                    .setContamination(contamination)
                                    .setSeed(seed)
                            )
        
        self.__fittedModel = None
        
    """
    Fits the Isolation Forest model using training data.
    
    """
    def train(self):
        self.__fittedModel = JavaTransformer._from_java(self.__model.fit(self.__df_train))
        return self.__fittedModel

    """
    Apply the fitted model to a new dataset (e.g., test set).
    
    """
    def test(self, model):
        if model is None:
            raise RuntimeError("Model must exist")
        
        return model.transform(self.__df_test)
    
    def evaluate(self, df_with_preds):

        df_eval = df_with_preds \
            .withColumn("pred_binary", when(col(self.__predictionCol) == -1, 1).otherwise(0)) \
            .withColumn("label_binary", when(col(self.__labelCol) == 0, 1).otherwise(0))

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label_binary",
            predictionCol="pred_binary",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(df_eval) * 100
        #print(f"Accuracy: {accuracy:.4f}")

        # Confusion Matrix
        cm = df_eval.groupBy("label_binary", "pred_binary").count().collect()
        matrix = {(row["label_binary"], row["pred_binary"]): row["count"] for row in cm}
        
        tn = matrix.get((0, 0), 0)
        fp = matrix.get((0, 1), 0)
        fn = matrix.get((1, 0), 0)
        tp = matrix.get((1, 1), 0)

        confusion_matrix = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

        """print("Confusion Matrix Summary:")
        print(f"TP (anomaly correctly detected):     {tp}")
        print(f"TN (normal correctly detected):      {tn}")
        print(f"FP (normal misclassified as anomaly): {fp}")
        print(f"FN (anomaly missed):                 {fn}")"""

        return accuracy, confusion_matrix