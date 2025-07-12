
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType, IntegerType

"""
This class is an implementation of the Isolation Forest algorithm from a JAR file resulted from building a GitHub project which
includes an implementation of Isolation Forest based on Spark computations. To create the models, we use, as inputs,
vectors that are the features from spark datasets, that are properly converted to an adequate format before being used
by the models

NOTE: this model is here only for testing, it will not be used because it can't be saved on MLFlow

Source: https://github.com/linkedin/isolation-forest (see README for all the instructions to build the JAR file before being used)

"""
class IsolationForest:

    """
    The class is initialized with the following parameters:
    
        spark_session: the Spark session used to load the JAR file that contains the used IF model
        
        df_train: the Spark dataframe used for model training
        
        df_test: the Spark dataframe used for model testing
        
        featuresCol: the name of the column in the Spark dataframes of train and test datasets that contains all the features of each
                    example of the dataset that are used by the model for training and testing

        labelCol: the name of the column in train and test datasets that corresponds to the label of the dataset. Since this is an
                    unsupervised algorithm, this "label" is only used in testing to compute the evaluation metrics that measure the efficacy
                    of the model during testing 

        predictionCol: the name of the column in train and test datasets that will contain the predictions of the model during testing

        maxSamples: the fraction of samples used to train each tree if between 0.0 and 1.0; otherwise, is the number of samples

        maxFeatures (default=1.0): the fraction of features used to train each tree if between 0.0 and 1.0; otherwise, is the number of samples
                                it's set to 1.0 to indicate that all features are used

        seed (default=42): a arbitrary number used for pseudo-randomness of the selection of the feature and split values
                             for each branching step and each tree in the forest, in the model training

        contamination (default=0.25): the expected fraction of outliers in the train dataset

        contaminationErrorRate (default=0.9): value that will be multiplied by contamination to determine an adequate value for contaminationError
                                            according to the size of the training dataset 

        numTrees: the number of trees used for training; it varies according to the dataset size

        N: the number of training samples
    
    """
    def __init__(self, spark_session, df_train, df_test, featuresCol,
                 predictionCol, labelCol, maxFeatures=1.0, seed=42, 
                 contamination=0.25, contaminationErrorRate=0.9):
        
        self.__spark_session = spark_session
        self.__df_train = df_train
        self.__df_test = df_test
        self.__predictionCol = predictionCol
        self.__featuresCol = featuresCol
        self.__labelCol = labelCol
        self.__N = df_train.count()
        self.__maxSamples = 0.3 if self.__N >= 50 else 1.0    # if N is between 15 and 49 samples, 1.0; otherwise, it's 0.3
        self.__numTrees = self.__set_num_trees(self.__N)
        
        # NOTE: uncomment this line to print the number of trees used for model training 
        #print("numTrees:", self.__numTrees)

        # Build Java IsolationForest Estimator
        # bootstrap set to False indicates that the sampling is not made with replacement, i.e., does not contain replied samples 
        self.__model = spark_session._jvm.com.linkedin.relevance.isolationforest.IsolationForest() \
                .setNumEstimators(self.__numTrees) \
                .setMaxSamples(self.__maxSamples) \
                .setBootstrap(False) \
                .setMaxFeatures(float(maxFeatures)) \
                .setFeaturesCol(featuresCol) \
                .setPredictionCol(predictionCol) \
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
        return self.__model.fit(self.__df_train.select(self.__featuresCol)._jdf)

    """
    Apply the fitted model to a new dataset (e.g., test set).
    
    """
    def predict(self, model, df):
        if model is None:
            raise RuntimeError("Model does not exist!")
        
        self.__df_test = df.drop(self.__predictionCol)._jdf
        
        java_df = model.transform(self.__df_test)
        return DataFrame(java_df, self.__spark_session)     # convert to a Spark dataframe, since this is an Java object

    """
    This method evaluates the predictions calculated by the model during testing, to give an idea of the model's efficacy
    As an argument, the method receives 'df_with_preds', which corresponds to a Spark dataframe that contains the model's calculated predictions
    It returns a dictionary that contains the confusion matrix, i.e., true negatives (TN), true positives (TP), false positives (FP) and false negatives (FN),
    and the report that contains the most relevant evaluation metrics

    """
    def evaluate(self, df_with_preds):

        evaluator = MulticlassClassificationEvaluator(
            labelCol=self.__labelCol,
            predictionCol=self.__predictionCol,
            metricName="accuracy"
        )

        accuracy = evaluator.evaluate(df_with_preds)

        self.__df_test = df_with_preds \
            .withColumn(self.__predictionCol, col(self.__predictionCol).cast(IntegerType()))

        # Confusion Matrix
        cm = self.__df_test.groupBy(self.__labelCol, self.__predictionCol).count().collect()
        matrix = {(row[self.__labelCol], row[self.__predictionCol]): row["count"] for row in cm}
        
        tn = matrix.get((0, 0), 0)
        fp = matrix.get((0, 1), 0)
        fn = matrix.get((1, 0), 0)
        tp = matrix.get((1, 1), 0)

        # Confusion matrix
        confusion_matrix = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

        # Metrics
        precision_class_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0        # Precision (class 1 -> anomaly)
        recall_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0           # Recall (class 1 -> anomaly)
        recall_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0           # Precision (class 0 -> normal)
        f1_score_class_1 = (2 * precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1) \
                                if (precision_class_1 + recall_class_1) > 0 else 0.0         # F1-Score (class 1 -> anomaly)

        # Dictionary with relevant evaluation metrics to analyse the efficacy of the model in testing
        report = {
            "Accuracy": accuracy,
            "Recall (class 1 -> anomaly)": recall_class_1,
            "Precision (class 1 -> anomaly)": precision_class_1,
            "F1-Score (class 1 -> anomaly)": f1_score_class_1,
            "Recall (class 0 -> normal)": recall_class_0
        }

        return confusion_matrix, report
  
    """
    This method corresponds to the test of the model. This method will call the predict method
    to calculate the predictions using the trained model, and then it will call the evaluate method to compute and return the evaluation metrics

    """
    def test(self, model):
        df_predictions = self.predict(model, self.__df_test)
        return self.evaluate(df_predictions)