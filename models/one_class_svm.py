from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when
import numpy as np
from sklearn.svm import OneClassSVM as OCSVM
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType

# TODO seems to be reasonably working, better than Isolation Forest: keep making improvements

class OneClassSVM:
    def __init__(self, spark_session, df_train, df_test, featuresCol, predictionCol, labelCol, intrusion_rate):
        self.__spark_session = spark_session
        self.__df_train = df_train
        self.__df_test = df_test
        self.__featuresCol = featuresCol
        self.__predictionCol = predictionCol
        self.__labelCol = labelCol
        self.__nu = intrusion_rate * 0.01       # TODO review this
        self.__features = np.array(self.__df_train.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())
        self.__gamma = 1 / self.__features.shape[1]

    def train(self):
        self.__model = OCSVM(kernel='rbf', nu=self.__nu, gamma=self.__gamma)
        return self.__model.fit(self.__features)
    
    def predict(self, model, features):
        return model.predict(features)

    def test(self, model):
        features = np.array(self.__df_test.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())
        preds = self.predict(model, features)

        # Convert scikit predictions to binary labels (-1: anomaly -> 1: normal -> 0)
        pred_labels = np.where(preds == -1, 1, 0)

        # Extrai os rótulos reais do DataFrame de teste
        true_labels = self.__df_test.select(self.__labelCol).rdd.map(lambda r: int(r[0])).collect()

        # Converte todos os elementos para tipos nativos do Python
        data = [(int(label), int(pred)) for label, pred in zip(true_labels, pred_labels)]

        # Cria DataFrame com os nomes das colunas
        pred_df = self.__spark_session.createDataFrame(data, [self.__labelCol, self.__predictionCol])
        return pred_df


    def evaluate(self, df_with_preds):
        """Avalia os resultados do modelo"""
        df_eval = df_with_preds.withColumn(self.__predictionCol, col(self.__predictionCol).cast(DoubleType()))

        # Accuracy
        evaluator = MulticlassClassificationEvaluator(
            labelCol=self.__labelCol, predictionCol=self.__predictionCol, metricName="accuracy"
        )
        accuracy = evaluator.evaluate(df_eval)

        # Confusion Matrix
        cm = df_eval.groupBy(self.__labelCol, self.__predictionCol).count().collect()
        matrix = {(row[self.__labelCol], row[self.__predictionCol]): row["count"] for row in cm}

        print(matrix)

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
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        report = {
            "Accuracy": accuracy,
            "Precision (anomaly)": precision,
            "Recall (anomaly)": recall,
            "F1-Score (anomaly)": f1_score
        }

        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        for k, v in confusion.items():
            print(f"{k}: {v}")
        print("Classification Report:")
        for k, v in report.items():
            print(f"{k}: {v:.4f}")

        return accuracy, {**confusion, **report}