

class KNNClassifier:

    """
    It initializes the KNN model.
    
    Parameters:
        k (int): The number of neighbors to consider.
        train_data (pyspark dataframe): The training data.
        test_data (pyspark dataframe): The test data.

    """
    def __init__(self, k, train_df, test_df, featuresCol, labelCol):
        self.__k = k
        self.__train_data = train_df.toPandas()
        self.__test_data = test_df.toPandas()
        self.__featuresCol = featuresCol
        self.__labelCol = labelCol

 
    """
    Predicts the label for a single observation.
    
    """
    def predict(self, observation):

        distances = []
        obs_features = observation[self.__featuresCol]

        for _, train_obs in self.__train_data.iterrows():
            train_features = train_obs[self.__featuresCol]
            distance = sum((a - b) ** 2 for a, b in zip(obs_features, train_features)) ** 0.5
            distances.append((distance, train_obs[self.__labelCol]))

        distances.sort(key=lambda x: x[0])
        nearest_labels = [label for _, label in distances[:self.__k]]
        prediction = max(set(nearest_labels), key=nearest_labels.count)

        return prediction

     
    """
    Evaluates the model on a labeled test dataset.
    Returns accuracy and total samples.
    
    """
    def test(self):

        correct = 0

        for _, test_obs in self.__test_data.iterrows():
            pred = self.predict(test_obs)
            true_label = test_obs[self.__labelCol]

            if pred == true_label:
                correct += 1

        total = len(self.__test_data)
        accuracy = correct / total if total > 0 else 0.0

        return accuracy