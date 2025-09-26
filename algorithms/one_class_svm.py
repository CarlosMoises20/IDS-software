
import numpy as np
from sklearn.svm import OneClassSVM as OCSVM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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
    def __init__(self, df_train, df_test, featuresCol, labelCol):
        self.__df_train = df_train
        self.__df_test = df_test
        self.__featuresCol = featuresCol
        self.__labelCol = labelCol

    """
    Method used to train the model, using the training dataset
    
    """
    def train(self):
        
        # convert the "features" column from the dataframe into a numpy array, the adequate format for sklearn models
        # this column results from assembling all attributes of each row into one vector with the appropriated format
        features = np.array(self.__df_train.select(self.__featuresCol).toPandas()[self.__featuresCol].tolist())

        # kernel='rbf' allows to learn non-linear relationships and detect rare outliers; there's no other solution for kernel
        # gamma = 'scale' allows the model to adapt to the data variance
        # 'tol': tolerance value that decides how small the gradient in the optimization algorithm has to be to consider that
        # it converged; a smaller 'tol' will result in a more accurate training, but also resource-demanding
        # NU is the upper bound of the contamination rate in the dataset
        # NU is also the lower bound of the fraction of support vectors
        model = OCSVM(tol=1e-12, nu=0.1)

        return model.fit(features)

    """
    Method used to test the model, using the testing dataset
    
    """
    def test(self, model):  
        # Use the model to predict the labels for the samples in the test dataset
        y_preds = self.predict(model)

        # Compute the evaluation metrics to determine the efficacy of the model during testing
        return self.evaluate(y_preds)

    """
    This method is used for the model to predict the labels of the testing samples,
    based on the learnt patterns during training
    
    """
    def predict(self, model):

        if model is None:
            raise Exception("Model must be specified!")

        # convert the "features" column from the dataframe into a numpy array, the adequate format for sklearn models
        # this column results from assembling all attributes of each row into one vector with the appropriated format
        features = np.array(self.__df_test.select(self.__featuresCol).toPandas()[self.__featuresCol].tolist())
        
        y_preds = model.predict(features)

        return [0 if pred == 1 else 1 for pred in y_preds]

    """
    Evaluates the model's results, computing the evaluation metrics to determine the efficacy of the model during testing
    
    """
    def evaluate(self, y_preds):

        # Real (expected) labels of the test dataset
        y_true = self.__df_test.select(self.__labelCol).rdd.map(lambda x: x[0]).collect()

        # Report with relevant evaluation metrics such as recall, f1-score and precision
        report = classification_report(y_true, y_preds, output_dict=True, zero_division=0)
        
        # Confusion matrix 
        tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel().tolist()
        conf_matrix = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

        # Accuracy: it represents the rate of correctly classified samples
        accuracy = accuracy_score(y_true, y_preds)
        
        return accuracy, conf_matrix, report