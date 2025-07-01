

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
        self.__k = max(5, min(round(df_train.count() * 0.01), 15))

    def train(self):
        print("k:", self.__k)
        model = LOFModel(n_neighbors=self.__k, 
                         n_jobs=-1,
                         novelty=True,
                         p=1)
        features = np.array(self.__df_train.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())
        return model.fit(features)

    def test(self, model):
        if self.__df_test is None:
            print("Test dataset is empty. Skipping testing.")
            return None, None, None
        y_pred = self.predict(model)
        return self.evaluate(y_pred)

    def predict(self, model):
        if model is None:
            raise Exception("Model must be created first!")
        features = np.array(self.__df_test.select(self.__featuresCol).rdd.map(lambda x: x[0]).collect())
        y_pred = model.predict(features)
        return np.array([0 if pred == 1 else 1 for pred in y_pred])
    
    def evaluate(self, y_pred):
        y_true = np.array(self.__df_test.select(self.__labelCol).rdd.map(lambda x: x[0]).collect()) 
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        conf_matrix = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy, conf_matrix, report