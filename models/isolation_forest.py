
from sklearn.ensemble import IsolationForest as IF
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

class IsolationForest:

    def __init__(self, df_train, df_test, featuresCol, labelCol):
        self.__X_train = np.array(df_train.select(featuresCol).rdd.map(lambda row: row[0]).collect())
        self.__X_test = np.array(df_test.select(featuresCol).rdd.map(lambda row: row[0]).collect())
        #print("X_train shape:", self.__X_train.shape)
        #print("X_test shape:", self.__X_test.shape)
        #self.__y_train = np.array(df_train.select(labelCol).rdd.map(lambda row: row[0]).collect())
        self.__y_test = np.array(df_test.select(labelCol).rdd.map(lambda row: row[0]).collect())
        self.__model = IF(random_state=42)
        

    def train(self):
        self.__model.fit(X=self.__X_train)
        return self.__model

    def test(self):
        # Previsão retorna -1 para anomalias e 1 para normais
        preds = self.__model.predict(self.__X_test)
        
        # Converte para 0 e 1, onde 1 = anomalia (intrusão)
        preds_binary = np.where(preds == -1, 1, 0)
        
        return preds_binary
    

    def evaluate(self):
        preds = self.test()
        accuracy = accuracy_score(self.__y_test, preds)
        cm = confusion_matrix(self.__y_test, preds)
        return accuracy, cm