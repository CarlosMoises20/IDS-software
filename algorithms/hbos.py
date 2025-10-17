
import numpy as np
import math
from pyod.models.hbos import HBOS as HBOSModel
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

"""
Histogram-Based Outlier Score; unsupervised method used to detect outliers in test dataset, in 
an approach based on histograms (bins)

"""
class HBOS:

    """
    Initializes the class with the following parameters
    
        df_train: train dataset
        df_test: test dataset
        featuresCol: name of the column in the dataframe with the features used by the model for training and testing
        labelCol: name of the column in the dataframe that corresponds to the label of the dataset, only used to compute
                    evaluation metrics in testing, since this is an unsupervised algorithm
    
    """
    def __init__(self, df_train, df_test, featuresCol, labelCol):
        self.__df_train = df_train
        self.__df_test = df_test
        self.__featuresCol = featuresCol
        self.__labelCol = labelCol

    """
    Train the model using the indicated contamination, that represents the outlier rate
    in the training dataset
    
    """
    def train(self):

        N = self.__df_train.count()         # the size of the training dataset
        
        # follow a often used rule, that is setting the number of bins to the square root of the number of training instances N
        # The rule is specified on this reference: https://www.goldiges.de/publications/HBOS-KI-2012.pdf
        num_bins = round(math.sqrt(N))

        # NOTE: uncomment this line to print the computed number of bins for the algorithm
        #print("number of bins:", num_bins)

        # num_bins is the number of used bins for each feature, alpha is a regularizer that prevents overflow
        # and tol is a parameter which decides the flexibility while dealing with the samples falling outside the bins
        # contamination is a parameter used to define the threshold in the decision function used to classify new instances,
        # and it represents the expected outlier rate in the dataset
        model = HBOSModel(n_bins=num_bins, tol=1e-20, contamination=1/3)
        
        features = np.array(self.__df_train.select(self.__featuresCol).toPandas()[self.__featuresCol].tolist())
        
        return model.fit(features)

    """
    Test the model and return the evaluation metrics
    
    """
    def test(self, model):
        Y_pred = self.predict(model)
        return self.evaluate(Y_pred)

    """
    Predict the labels using the indicated model
    
    """
    def predict(self, model):
        # convert dataset to pandas to be compatible with pyod-based HBOS model 
        features = np.array(self.__df_test.select(self.__featuresCol).toPandas()[self.__featuresCol].tolist())
        return model.predict(features)

    """
    Evaluates the model's results, using the predicted labels and the real labels
    of the test dataset

    As an argument, it receives the predictions (Y_pred) in pandas format, and it returns:

        confusion_matrix: true positives (tp), true negatives (tn), false positives (fp), false negatives (fn)
        accuracy: indicates how close predictions are from the real / expected labels
        report: contains more relevant evaluation metrics such as F1-Score, Recall and Precision
    
    """
    def evaluate(self, Y_pred):
        
        # Extract the real / expected labels of the test dataset
        Y_true = self.__df_test.select(self.__labelCol).rdd.map(lambda x: x[0]).collect()     

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true=Y_true, y_pred=Y_pred).ravel().tolist()
        conf_matrix = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
        
        # Accuracy
        accuracy = accuracy_score(Y_true, Y_pred)
        
        # Report with some relevant evaluation metrics
        report = classification_report(Y_true, Y_pred, output_dict=True, zero_division=0)
        
        return accuracy, conf_matrix, report