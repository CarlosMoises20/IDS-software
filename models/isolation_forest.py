
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType, IntegerType
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import IsolationForest as IF
import numpy as np
from pyspark.sql import DataFrame


class IsolationForest:

    def __init__(self, df_train, df_test, featuresCol,
                 labelCol, seed=42):
        
        self.__df_train = df_train.select(featuresCol, labelCol)
        self.__df_test = df_test.select(featuresCol, labelCol)
        self.__featuresCol = featuresCol
        self.__labelCol = labelCol
        self.__numTrees = self.__set_num_trees(df_train.count())
        print("numTrees:", self.__numTrees)

        # Build Java IsolationForest Estimator
        self.__model = IF(n_estimators=self.__numTrees, 
                          n_jobs=-1,
                          random_state=seed)

    """
    This function adjusts the number of trees in training according to the size of the training dataset
    The larger the training dataset, the larger the number of trees, to maintain the efficacy of the model
    
    """
    def __set_num_trees(self, num_training_samples):
        return min(10000 + num_training_samples * 2, 30000)

    """
    Fits the Isolation Forest model using training data.
    
    """
    def train(self):
        features = np.array(self.__df_train.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())
        return self.__model.fit(features)

    """
    Apply the fitted model to a new dataset (e.g., test set).
    
    """
    def predict(self, model):
        if model is None:
            raise RuntimeError("Model does not exist!")
        features = np.array(self.__df_test.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())
        y_pred = model.predict(features)
        return np.array([0 if pred == 1 else 1 for pred in y_pred])
    

    def evaluate(self, y_pred):
        y_true = np.array(self.__df_test.select(self.__labelCol).rdd.map(lambda x: x[0]).collect()) 
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy, conf_matrix, report
  
    def test(self, model):

        if self.__df_test is None:
            print("Test dataset is empty. Skipping testing.")
            return None, None

        y_pred = self.predict(model)
        return self.evaluate(y_pred)


class IsolationForestLinkedIn:

    def __init__(self, spark_session, df_train, df_test, featuresCol,
                 scoreCol, predictionCol, labelCol,
                 maxSamples=0.3, maxFeatures=1.0, seed=42, contamination=0.15):
        
        self.__spark_session = spark_session
        self.__df_train = df_train.drop("prediction", "score")
        self.__df_test = df_test.drop("prediction", "score")
        self.__predictionCol = predictionCol
        self.__featuresCol = featuresCol
        self.__scoreCol = scoreCol
        self.__labelCol = labelCol
        self.__numTrees = self.__set_num_trees(df_train.count())
        print("numTrees:", self.__numTrees)

        # Build Java IsolationForest Estimator
        self.__model = spark_session._jvm.com.linkedin.relevance.isolationforest.IsolationForest() \
                .setNumEstimators(self.__numTrees) \
                .setMaxSamples(maxSamples) \
                .setBootstrap(False) \
                .setMaxFeatures(float(maxFeatures)) \
                .setFeaturesCol(featuresCol) \
                .setPredictionCol(predictionCol) \
                .setScoreCol(scoreCol) \
                .setRandomSeed(seed) \
                .setContamination(contamination) \
                .setContaminationError(contamination * 0.5)

    """
    This function adjusts the number of trees in training according to the size of the training dataset
    The larger the training dataset, the larger the number of trees, to maintain the efficacy of the model
    
    """
    def __set_num_trees(self, num_training_samples):
        return min(1500 + num_training_samples, 2500)

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

        evaluator = MulticlassClassificationEvaluator(
            labelCol=self.__labelCol,
            predictionCol=self.__predictionCol,
            metricName="accuracy"
        )

        accuracy = evaluator.evaluate(df_with_preds)

        self.__df_test = df_with_preds \
            .withColumn(self.__predictionCol, col(self.__predictionCol).cast(IntegerType())) \
            .withColumn(self.__scoreCol, col(self.__scoreCol).cast(DoubleType()))

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