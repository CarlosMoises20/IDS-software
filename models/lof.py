

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import LocalOutlierFactor as LOFModel
import numpy as np

"""
Local Outlier Factor; unsupervised method used to detect outliers in test dataset, in 
an approach inspired on kNN

"""
class LOF:

    """
    Initializes the class with the following parameters
    
        df_train: Spark dataframe which corresponds to the train dataset
        
        df_test: Spark dataframe which corresponds to the test dataset
        
        featuresCol: name of the column in the dataframe with the features used by the model for training and testing
        
        labelCol: name of the column in the dataframe that corresponds to the label of the dataset, only used to compute
                    evaluation metrics in testing, since this is an unsupervised algorithm

        k: the chosen 'k' for LOF according to the size of the training dataset
    
    """
    def __init__(self, df_train, df_test, featuresCol, labelCol):
        self.__df_train = df_train
        self.__df_test = df_test
        self.__featuresCol = featuresCol
        self.__labelCol = labelCol
        self.__k = max(5, min(round(df_train.count() * 0.01), 15))

    """
    Fits the Local Outlier Factor model using training data.
    
    """
    def train(self):
        
        # NOTE uncomment this line to print 'k', the number of the nearest neighbors
        #print("k:", self.__k)

        # n_neighbors is the number of neighbors used to calculate the distance from the current point to those neighbors
        # n_jobs set to -1 indicates that the number of jobs running in parallel is the number of the processors of the machine, which
        # speeds up the process
        # novelty set to True indicates that we want to detect specially unknown anomalies. This assumes that the train dataset has no anomalies
        # 'p' set to 1 means that the metric used to calculate the distance between the points is the Manhattan distance 
        model = LOFModel(n_neighbors=self.__k, 
                         n_jobs=-1,
                         novelty=True,
                         p=1)
        
        # convert the assembled features from the Spark dataset into an adequate format for the sklearn-based LOF model
        features = np.array(self.__df_train.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())

        # Fit the model using the assembled features after being converted
        return model.fit(features)

    """
    This method corresponds to the test of the model. This method will call the predict method
    to calculate the predictions using the trained model, and then it will call the evaluate method to compute and return the evaluation metrics

    """
    def test(self, model):
        y_pred = self.predict(model)
        return self.evaluate(y_pred)

    """
    Apply the fitted model to a new dataset (e.g., test set).
    Returns the predictions that are calculated by the model based on what it learned during training
    
    """
    def predict(self, model):
        if model is None:
            raise Exception("Model must be created first!")
        
        # convert the assembled features from the Spark dataset into an adequate format for the sklearn-based LOF model
        features = np.array(self.__df_test.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())

        # calculate the predictions based on the assembled and processed features
        y_pred = model.predict(features)
        
        # sklearn LOF predictions are 1 for inliers and -1 for outliers; but in this case, we want to return 0 for inliers
        # and 1 for outliers
        return np.array([0 if pred == 1 else 1 for pred in y_pred])
    
    """
    This method evaluates the predictions calculated by the model during testing, to give an idea of the model's efficacy
    As an argument, the method receives 'y_pred', which corresponds to a numpy array that contains the model's calculated predictions
    It returns the confusion matrix, a dictionary with true negatives (tn), true positives (tp), false positives (fp) and false negatives (fn),
    the accuracy and the report that contains all resumed evaluation metrics

    """
    def evaluate(self, y_pred):
        # Real (expected) labels of the test dataset
        y_true = np.array(self.__df_test.select(self.__labelCol).rdd.map(lambda x: x[0]).collect()) 

        # Report with relevant evaluation metrics such as recall, f1-score and precision
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)       

        # Confusion matrix 
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        conf_matrix = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

        # Accuracy: it represents the rate of correctly classified samples
        accuracy = accuracy_score(y_true, y_pred)
        
        return accuracy, conf_matrix, report