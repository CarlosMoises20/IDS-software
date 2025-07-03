
from pyspark.sql.functions import col, monotonically_increasing_id
import numpy as np
from sklearn.svm import OneClassSVM as OCSVM
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType


"""
This class corresponds to an implementation of One-Class SVM that takes advantage of the sklearn OCSVM model
that uses, as input, datasets from Spark that are converted to adequate formats to be processed by the model
The One-Class SVM algorithm is one of the most used algorithms for unsupervised outlier detection. 
It assumes that, even though the model is trained with normal traffic, the training dataset might contain some outliers,
trying to ignore those deviant observations

"""
class OneClassSVM:

    """
    The class is initialized with the following parameters

        spark_session: the Spark session that is used to create a Spark dataframe with the model's predictions
                        to be returned for evaluation

        df_train: the Spark dataframe that corresponds to the dataset used for training in OCSVM
        df_test: the Spark dataframe that corresponds to the dataset used for testing in OCSVM
        featuresCol: the name of the column on the Spark dataframes corresponding to the training / testing dataset
                    that contains the features used by the models for training and testing
        
        labelCol: the name of the column on the Spark dataframe that corresponds to the label. This is an unsupervised 
                algorithm, so the label will only be used for testing, to compute the evaluation metrics to determine the
                efficacy of the model in testing

    """
    def __init__(self, spark_session, df_train, df_test, featuresCol, 
                 predictionCol, labelCol):
        
        self.__spark_session = spark_session
        self.__df_train = df_train
        self.__df_test = df_test
        self.__featuresCol = featuresCol
        self.__predictionCol = predictionCol
        self.__labelCol = labelCol

    """
    Method used to train the model, using the training dataset
    
    """
    def train(self):
        
        # convert the "features" column from the dataframe into a numpy array, the adequate format for sklearn models
        # this column results from assembling all attributes of each row into one vector with the appropriated format
        features = np.array(self.__df_train.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())
        
        N = self.__df_train.count()     # the number of training samples

        # NU is the upper bound of the contamination rate in the dataset, that normally is smaller in higher datasets
        # NU is also the lower bound of the fraction of support vectors
        # ensure nu is always between min_nu and max_nu, and is higher in smaller datasets
        # the higher the N, the larger the dataset and the lower the 'nu' is expected to be
        # M controls the scale: higher M means that 'nu' starts higher for small N and decreases more slowly as N increases
        nu = max(0.0005, min(0.1, 15 / N)) if N >= 20 else max(0.1, min(0.5, 4 / N))

        print("nu:", nu)

        # 'rbf' allows to learn non-linear relationships and detect rare outliers; there's no other solution for kernel
        # gamma = 'scale' allows the model to adapt to the data variance
        # NOTE it works better with large datasets, for smaller datasets, a too large NU loses too much anomalies and a
        # too small NU gives too much false positives
        # 'tol': tolerance value that decides how small the gradient in the optimization algorithm has to be to consider that
        # it converged; a smaller 'tol' will result in a more accurate training, but also slower
        self.__model = OCSVM(tol=1e-10, nu=nu)

        return self.__model.fit(features)

    """
    Method used to test the model, using the testing dataset
    
    """
    def test(self, model):

        # If the testing model does not exist (for not having sufficient samples for testing), testing will be skipped
        if self.__df_test is None:
            print("Test dataset is empty. Skipping testing.")
            return None, None, None
        
        # Use the model to predict the labels for the samples in the test dataset
        df_preds = self.predict(model)

        # Compute the evaluation metrics to determine the efficacy of the model during testing
        accuracy, evaluation = self.evaluate(df_preds)

        return accuracy, evaluation, df_preds

    """
    This method is used for the model to predict the labels of the testing samples,
    based on the learnt patterns during training
    
    """
    def predict(self, model):

        if model is None:
            raise Exception("Model must be specified!")

        df_test_indexed = self.__df_test.withColumn("row_id", monotonically_increasing_id())

        # convert the "features" column from the dataframe into a numpy array, the adequate format for sklearn models
        # this column results from assembling all attributes of each row into one vector with the appropriated format
        features = np.array(df_test_indexed.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())
        
        preds = model.predict(features)

        # Convert scikit predictions to binary labels (-1: anomaly -> 1: normal -> 0)
        pred_labels = np.where(preds == -1, 1, 0)

        # Extracts the real test labels from the dataframe
        true_labels = df_test_indexed.select(self.__labelCol).rdd.map(lambda r: int(r[0])).collect()

        row_ids = df_test_indexed.select("row_id").rdd.map(lambda r: r[0]).collect()

        data = [(row_id, int(label), int(pred)) for row_id, label, pred in zip(row_ids, true_labels, pred_labels)]

        # Create a dataframe with the columns' names and the predictions
        pred_df = self.__spark_session.createDataFrame(data, ["row_id", self.__labelCol, self.__predictionCol])

        # Dataframe that results from the join of the created dataframe and the original indexed test dataframe 
        joined_df = df_test_indexed.drop(self.__predictionCol, self.__labelCol) \
                                    .join(pred_df, on="row_id", how="inner")

        return joined_df.drop("row_id")

    """
    Evaluates the model's results, computing the evaluation metrics to determine the efficacy of the model during testing
    
    """
    def evaluate(self, df_with_preds):
        
        # Convert the prediction column to DoubleType to be supported by the Spark evaluation "MulticlassClassificationEvaluator"
        df_eval = df_with_preds.withColumn(self.__predictionCol, col(self.__predictionCol).cast(DoubleType()))

        # Accuracy
        evaluator = MulticlassClassificationEvaluator(
            labelCol=self.__labelCol, predictionCol=self.__predictionCol, metricName="accuracy"
        )

        # Compute the accuracy
        accuracy = evaluator.evaluate(df_eval)

        # Build confusion matrix from prediction and label columns of the spark dataframe corresponding to the test dataset
        cm = df_eval.groupBy(self.__labelCol, self.__predictionCol).count().collect()
        matrix = {(row[self.__labelCol], row[self.__predictionCol]): row["count"] for row in cm}

        tn = matrix.get((0, 0), 0)      # True negatives: normal samples correctly detected as normal
        fp = matrix.get((0, 1), 0)      # False positives: normal samples incorrectly detected as anomalies
        fn = matrix.get((1, 0), 0)      # False negatives: anomalies incorrectly detected as normal samples
        tp = matrix.get((1, 1), 0)      # True positives: anomalies correctly detected as anomalies

        # Confusion matrix in dictionary format
        confusion = {
            "TP (anomaly detected)": tp,
            "TN (normal detected)": tn,
            "FP (normal as anomaly)": fp,
            "FN (missed anomaly)": fn
        }

        # Metrics
        precision_class_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0        # Precision (class 1 -> anomaly)
        recall_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0           # Recall (class 1 -> anomaly)
        recall_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0           # Precision (class 0 -> normal)
        f1_score_class_1 = (2 * precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1) \
                                if (precision_class_1 + recall_class_1) > 0 else 0.0         # F1-Score (class 1 -> anomaly)

        # Dictionary with relevant evaluation metrics to analyse the efficacy of the model in testing
        report = {
            "Accuracy": accuracy,
            "Precision (class 1 -> anomaly)": precision_class_1,
            "Recall (class 1 -> anomaly)": recall_class_1,
            "Recall (class 0 -> normal)": recall_class_0,
            "F1-Score (class 1 -> anomaly)": f1_score_class_1
        }

        return accuracy, {**confusion, **report}