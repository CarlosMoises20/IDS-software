
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import IsolationForest as IF
import numpy as np


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
        
        # NOTE: uncomment to print the number of trees for model training
        #print("numTrees:", self.__numTrees)

        # Build Java IsolationForest Estimator
        # n_neighbors is the number of neighbors used to calculate the distance from the current point to those neighbors
        # n_jobs set to -1 indicates that the number of jobs running in parallel is the number of the processors of the machine, which
        # speeds up the process
        # random_state is the seed
        self.__model = IF(n_estimators=self.__numTrees, n_jobs=-1, random_state=seed)

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
    This method corresponds to the test of the model. This method will call the predict method
    to calculate the predictions using the trained model, and then it will call the evaluate method to compute and return the evaluation metrics

    """
    def test(self, model):
        y_pred = self.predict(model)
        return self.evaluate(y_pred)
