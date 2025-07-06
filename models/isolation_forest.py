
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType, IntegerType
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import IsolationForest as IF
import numpy as np
from pyspark.sql import DataFrame

"""
This class is an implementation of the Isolation Forest algorithm from sklearn library. To create the models, we use, as inputs,
vectors that are the features from spark datasets, that are properly converted to an adequate format before being used
by sklearn models

"""
class IsolationForest:

    """
    To initialize the class, the following parameters are passed as arguments
    
        df_train: the Spark dataframe that corresponds to the train dataset
        df_test: the Spark dataframe that corresponds to the test dataset
        featuresCol: the name of the column in train and test datasets that contains all dataset features assembled in Vector format, and
                     used for model training and testing  

        labelCol: the name of the column in train and test datasets that corresponds to the label of the dataset. Since this is an
                    unsupervised algorithm, this "label" is only used in testing to compute the evaluation metrics that measure the efficacy
                    of the model during testing

        seed (default=42): a arbitrary number used for pseudo-randomness of the selection of the feature and split values
                             for each branching step and each tree in the forest, in the model training
    
    """
    def __init__(self, df_train, df_test, featuresCol, labelCol, seed=42):
        self.__df_train = df_train
        self.__df_test = df_test
        self.__featuresCol = featuresCol
        self.__labelCol = labelCol
        self.__numTrees = self.__set_num_trees(df_train.count())
        
        # NOTE: uncomment to print the number of trees for the model training
        #print("numTrees:", self.__numTrees)

        # Build Java IsolationForest Estimator
        self.__model = IF(n_estimators=self.__numTrees, 
                          n_jobs=-1,
                          random_state=seed)

    """
    This function adjusts the number of trees in training according to the size of the training dataset
    The larger the training dataset, the larger the necessary number of trees to maintain
    the efficacy of the model. We need to be careful to choose the adequate number of trees, because too few 
    trees make the model to not learn all patterns, and too many trees make the model too overfitted and takes
    too much time to be generated, making it less efficient
    
    """
    def __set_num_trees(self, num_training_samples):
        return min(100 + int(num_training_samples // 5), 7000)

    """
    Fits the Isolation Forest model using training data.
    
    """
    def train(self):
        # Converts the dataset features into an adequate format for the IF model (a numpy array)
        features = np.array(self.__df_train.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())
        return self.__model.fit(features)

    """
    Apply the fitted model to a new dataset (e.g., test set).
    Returns the predictions that are calculated by the model based on what it learned during training
    
    """
    def predict(self, model):
        if model is None:
            raise RuntimeError("Model does not exist!")
        # Converts the dataset features into an adequate format for the IF model (a numpy array)
        features = np.array(self.__df_test.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())
        y_pred = model.predict(features)
        return np.array([0 if pred == 1 else 1 for pred in y_pred])
    
    """
    This method evaluates the predictions calculated by the model during testing, to give an idea of the model's efficacy
    As an argument, the method receives 'y_pred', which corresponds to a numpy array that contains the model's calculated predictions
    It returns the confusion matrix, a dictionary with true negatives (tn), true positives (tp), false positives (fp) and false negatives (fn),
    the accuracy and the report that contains all resumed evaluation metrics

    """
    def evaluate(self, y_pred):
        y_true = np.array(self.__df_test.select(self.__labelCol).rdd.map(lambda x: x[0]).collect()) 
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
        return accuracy, conf_matrix, report
  
    """
    This method corresponds to the test of the model. If the test dataset is empty (which might happen when the device contains very few
    samples and all those samples are used for training only), this phase will be skipped. Otherwise, this method will call the predict method
    to calculate the predictions using the trained model, and then it will call the evaluate method to compute and return the evaluation metrics

    """
    def test(self, model):

        if self.__df_test is None:
            print("Test dataset is empty. Skipping testing.")
            return None, None, None

        y_pred = self.predict(model)
        return self.evaluate(y_pred)


## TODO this seems to have better results, but verify it later
class IsolationForestLinkedIn:

    def __init__(self, spark_session, df_train, df_test, featuresCol,
                 scoreCol, predictionCol, labelCol,
                 maxSamples=0.3, maxFeatures=1.0, seed=42, contamination=0.25,
                 contaminationErrorRate=0.9):
        
        self.__spark_session = spark_session
        self.__df_train = df_train
        self.__df_test = df_test if df_test is not None else None
        self.__predictionCol = predictionCol
        self.__featuresCol = featuresCol
        self.__scoreCol = scoreCol
        self.__labelCol = labelCol
        self.__numTrees = self.__set_num_trees(df_train.count())
        
        # NOTE: uncomment this line to print the number of trees used for model training 
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
                .setContaminationError(contamination * contaminationErrorRate)

    """
    This function adjusts the number of trees in training according to the size of the training dataset
    The larger the training dataset, the larger the number of trees, to maintain the efficacy and efficiency of the model
    
    """
    def __set_num_trees(self, num_training_samples):
        return min(100 + int(num_training_samples // 5), 1800)

    """
    Fits the Isolation Forest model using training data.
    
    """
    def train(self):
        self.__df_train = self.__df_train.drop(self.__predictionCol, self.__scoreCol)
        return self.__model.fit(self.__df_train.select(self.__featuresCol, self.__labelCol)._jdf)

    """
    Apply the fitted model to a new dataset (e.g., test set).
    
    """
    def predict(self, model, df):
        if model is None:
            raise RuntimeError("Model does not exist!")
        
        self.__df_test = df.drop(self.__predictionCol, self.__scoreCol)._jdf
        
        java_df = model.transform(self.__df_test)
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

        # Metrics
        precision_class_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0        # Precision (class 1 -> anomaly)
        recall_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0           # Recall (class 1 -> anomaly)
        recall_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0           # Precision (class 0 -> normal)
        f1_score_class_1 = (2 * precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1) \
                                if (precision_class_1 + recall_class_1) > 0 else 0.0         # F1-Score (class 1 -> anomaly)

        # Dictionary with relevant evaluation metrics to analyse the efficacy of the model in testing
        report = {
            "Accuracy": f'{round(accuracy * 100, 2)}%',
            "Recall (class 1 -> anomaly)": f'{round(recall_class_1 * 100, 2)}%',
            "Precision (class 1 -> anomaly)": f'{round(precision_class_1 * 100, 2)}%',
            "F1-Score (class 1 -> anomaly)": f'{round(f1_score_class_1 * 100, 2)}%',
            "Recall (class 0 -> normal)": f'{round(recall_class_0 * 100, 2)}%'
        }

        return {**confusion_matrix, **report}, df_with_preds
  
    def test(self, model):
        df_predictions = self.predict(model, self.__df_test)
        return self.evaluate(df_predictions)