
from pyspark.sql.functions import col, monotonically_increasing_id
import numpy as np
from sklearn.svm import OneClassSVM as OCSVM
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType

# TODO write comments to explain each parameter on init

class OneClassSVM:
    def __init__(self, spark_session, df_train, df_test, featuresCol, 
                 predictionCol, labelCol):
        
        self.__spark_session = spark_session
        self.__df_train = df_train
        self.__df_test = df_test
        self.__featuresCol = featuresCol
        self.__predictionCol = predictionCol
        self.__labelCol = labelCol

    def train(self, nu=0.1):
        
        features = np.array(self.__df_train.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())
        
        N = self.__df_train.count()

        # ensure nu is always between min_nu and max_nu, and is higher in smaller datasets
        # the higher the N, the larger the dataset and the lower the 'nu' is expected to be
        # M controls the scale: higher M means that 'nu' starts higher for small N and decreases more slowly as N increases
        nu = max(0.0005, min(0.1, 15 / N)) if N > 20 else max(0.1, min(0.5, 12.5 / N))

        print("nu:", nu)

        # 'rbf' allows to learn non-linear relationships and detect rare outliers; there's no other solution for kernel
        # gamma = 'scale' allows the model to adapt to the data variance
        # NOTE it works better with large datasets, for smaller datasets, a too large NU loses too much anomalies and a
        # too small NU gives too much false positives
        self.__model = OCSVM(tol=1e-6, nu=nu)

        return self.__model.fit(features)

    def test(self, model):
        if self.__df_test is None:
            print("Test dataset is empty. Skipping testing.")
            return None, None, None
        df_preds = self.predict(model)
        accuracy, evaluation = self.evaluate(df_preds)
        return accuracy, evaluation, df_preds


    def predict(self, model):

        df_test_indexed = self.__df_test.withColumn("row_id", monotonically_increasing_id())

        features = np.array(df_test_indexed.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())
        
        preds = model.predict(features)

        # Convert scikit predictions to binary labels (-1: anomaly -> 1: normal -> 0)
        pred_labels = np.where(preds == -1, 1, 0)

        # Extracts the real test labels from the dataframe
        true_labels = df_test_indexed.select(self.__labelCol).rdd.map(lambda r: int(r[0])).collect()

        row_ids = df_test_indexed.select("row_id").rdd.map(lambda r: r[0]).collect()

        data = [(row_id, int(label), int(pred)) for row_id, label, pred in zip(row_ids, true_labels, pred_labels)]

        # Create a dataframe with the columns' names
        pred_df = self.__spark_session.createDataFrame(data, ["row_id", self.__labelCol, self.__predictionCol])

        joined_df = df_test_indexed.drop(self.__predictionCol, self.__labelCol) \
                                    .join(pred_df, on="row_id", how="inner")

        return joined_df.drop("row_id")

    """
    Evaluates the model's results
    
    """
    def evaluate(self, df_with_preds):
        
        df_eval = df_with_preds.withColumn(self.__predictionCol, col(self.__predictionCol).cast(DoubleType()))

        # Accuracy
        evaluator = MulticlassClassificationEvaluator(
            labelCol=self.__labelCol, predictionCol=self.__predictionCol, metricName="accuracy"
        )
        accuracy = evaluator.evaluate(df_eval)

        # Confusion Matrix
        cm = df_eval.groupBy(self.__labelCol, self.__predictionCol).count().collect()
        matrix = {(row[self.__labelCol], row[self.__predictionCol]): row["count"] for row in cm}

        tn = matrix.get((0, 0), 0)
        fp = matrix.get((0, 1), 0)
        fn = matrix.get((1, 0), 0)
        tp = matrix.get((1, 1), 0)

        confusion = {
            "TP (anomaly detected)": tp,
            "TN (normal detected)": tn,
            "FP (normal as anomaly)": fp,
            "FN (missed anomaly)": fn
        }

        # Metrics
        precision_class_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score_class_1 = (2 * precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1) \
                                if (precision_class_1 + recall_class_1) > 0 else 0.0

        report = {
            "Accuracy": accuracy,
            "Precision (anomaly)": precision_class_1,
            "Recall (anomaly)": recall_class_1,
            "Recall (normal)": recall_class_0,
            "F1-Score (anomaly)": f1_score_class_1
        }

        return accuracy, {**confusion, **report}