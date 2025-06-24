

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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
    def __init__(self, spark_session, df_train, df_test, featuresCol, predictionCol, labelCol, k=10):
        self.__spark_session = spark_session
        self.__df_train = df_train.select(featuresCol).toPandas()
        self.__df_test = df_test.select(featuresCol, labelCol).toPandas()
        self.__labelCol = labelCol
        self.__predictionCol = predictionCol
        self.__model = spark_session._jvm.org.apache.spark.ml.outlier.LOF() \
                                        .setFeaturesCol(featuresCol) \
                                        .setMinPts(k) \
                                        .setPredictionCol(predictionCol)


    def train(self):
        return self.__model.fit(self.__df_train._jdf)

    def test(self, model):
        df_preds = self.predict(model)
        return self.evaluate(df_preds)

    def predict(self, model):
        if model is None:
            raise Exception("Model must be created first!")
        
        jdf = model.transform(self.__df_test._jdf)
        return self.__spark_session.createDataFrame(jdf)
    
    def evaluate(self, df_preds):
        pdf = df_preds.select(self.__predictionCol, self.__labelCol).toPandas()
        y_true = pdf[self.__labelCol]
        y_pred = pdf[self.__predictionCol]
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy, conf_matrix, report