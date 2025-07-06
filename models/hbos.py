
import numpy as np
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
        N = self.__df_train.count()
        M = 35
        min_contamination = 0.01
        max_contamination = 0.2
        
        # ensure contamination is always between min_contamination and max_contamination, and is higher in smaller datasets
        # the higher the N, the larger the dataset and the lower the contamination is expected to be
        # M controls the scale: higher M means contamination starts higher for small N and decreases more slowly as N grows
        contamination = max(min_contamination, min(max_contamination, M / N))
        
        # NOTE: uncomment this line to print the expected outlier rate on the training dataset
        print("contamination:", contamination)

        model = HBOSModel(contamination=contamination, n_bins=10)
        
        df_train = self.__df_train.select(self.__featuresCol).toPandas()
        
        X = np.array(df_train[self.__featuresCol].tolist())
        
        return model.fit(X)

    """
    Test the model and return the evaluation metrics
    
    """
    def test(self, model):
        Y_pred = self.predict(model)
        accuracy, matrix, report = self.evaluate(Y_pred)
        return accuracy, matrix, report

    """
    Predict the labels using the indicated model
    
    """
    def predict(self, model):
        self.__df_test = self.__df_test.select(self.__featuresCol, self.__labelCol).toPandas()
        Y = np.array(self.__df_test[self.__featuresCol].tolist())
        return model.predict(Y)

    """
    Evaluates the model's results, using the predicted labels and the real labels
    of the test dataset
    
    """
    def evaluate(self, Y_pred):
        Y_true = self.__df_test[self.__labelCol].values
        tn, fp, fn, tp = confusion_matrix(y_true=Y_true, y_pred=Y_pred).ravel()
        conf_matrix = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
        accuracy = accuracy_score(Y_true, Y_pred)
        report = classification_report(Y_true, Y_pred, output_dict=True, zero_division=0)
        return accuracy, conf_matrix, report