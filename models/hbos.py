
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
        self.__df_train = df_train.select(featuresCol).toPandas()
        self.__df_test = df_test.select(featuresCol, labelCol).toPandas()
        self.__featuresCol = featuresCol
        self.__labelCol = labelCol

    """Train the model using the indicated contamination, that represents the outlier rate
    in the training dataset
    
    """
    def train(self, contamination, n_bins=10):
        model = HBOSModel(contamination=contamination, n_bins=n_bins)
        X = np.array(self.__df_train[self.__featuresCol].tolist())
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
        Y = np.array(self.__df_test[self.__featuresCol].tolist())
        return model.predict(Y)

    """
    Evaluates the model's results, using the predicted labels and the real labels
    of the test dataset
    
    """
    def evaluate(self, Y_pred):
        Y_true = self.__df_test[self.__labelCol].values
        cm = confusion_matrix(Y_true, Y_pred)
        accuracy = accuracy_score(Y_true, Y_pred)
        report = classification_report(Y_true, Y_pred, output_dict=True)
        return accuracy, cm, report