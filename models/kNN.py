
from models.functions import euclidean_dist


from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.param.shared import *
from pyspark.mllib.common import inherit_doc
from pyspark import keyword_only
from models.functions import euclidean_dist




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
        obs_features = getattr(observation, self.__featuresCol)

        for train_obs in self.__train_data:
            train_features = getattr(train_obs, self.__featuresCol)
            distance = sum((a - b) ** 2 for a, b in zip(obs_features, train_features)) ** 0.5
            distances.append((distance, getattr(train_obs, self.__labelCol)))

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

        for obs in self.__test_data:
            pred = self.predict(obs)
            true_label = obs[self.__labelCol]

            if pred == true_label:
                correct += 1

        total = len(self.__test_data)
        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "total_samples": total
        }



"""def kNN_apply(train_data, test_data, k_neighbors=10):
    
    # Collect training data ONCE
    train_data = train_data.collect()

    # (Optional) Collect test data ONCE too
    test_data = test_data.collect()

    acc = 0
    for obs in test_data:
        pred = kNN(k=k_neighbors, observation=obs, train_data_list=train_data)
        if pred == obs["intrusion"]:
            acc += 1

    accuracy_knn = acc / len(test_data)

    return accuracy_knn"""



"""def kNN(k, observation, train_data_list):
    #k: number of nearest neighbors
    #observation: the test observation (a Row object)
    #train_data_list: list of training examples (already collected from Spark)
    

    distances = []

    # Assuming your features are inside a vector column called "features"
    obs_features = observation["features"]

    for train_obs in train_data_list:
        train_features = train_obs["features"]

        # Here you need to compute the distance between the two feature vectors
        # Let's assume features are dense arrays, we can use simple Euclidean distance
        distance = sum((a - b) ** 2 for a, b in zip(obs_features, train_features)) ** 0.5

        distances.append((distance, train_obs["intrusion"]))  # assuming 'intrusion' is the label

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Get the labels of the k nearest neighbors
    nearest_labels = [label for _, label in distances[:k]]

    # Majority vote
    prediction = max(set(nearest_labels), key=nearest_labels.count)

    return prediction"""



"""# define knn algorithm
def KNN(k, observation, train_data):
    dists = []
    classes = []
    for i in range(train_data.count()):
        dist = euclidean_dist(train_data.collect()[i], observation)
        if len(dists) <= k:
            dists.append(dist)
            classes.append(train_data.collect()[i][len(observation)-1])
        else:
            for d in range(len(dists)):
                if dist < dists[d]:
                    del dists[d] 
                    del classes[d]
                    dists.append(dist)
                    classes.append(train_data.collect()[i][len(observation)-1])
    poss_classes = []
    class_counts = []
    for c in classes:
        if c not in poss_classes:
            poss_classes.append(c)
            class_counts.append(1)
        else:
            class_idx = poss_classes.index(c)
            class_counts[class_idx] += 1
    max_class = poss_classes[0]
    max_count = class_counts[0]
    for i in range(len(poss_classes)):
        if max_count < class_counts[i]:
            max_class = poss_classes[i]
            max_count = class_counts[i]
    return max_class"""



"""# get accuracy of classifier
def accuracy(train_data, test_data, k):
    acc = 0
    preds = []
    actuals = []
    for obs in range(test_data.count()):
        pred = KNN(k, test_data.collect()[obs], train_data)
        if pred == test_data.collect()[obs][len(test_data.collect()[0])-1]:
            acc += 1
        preds.append(pred)
        actuals.append(test_data.collect()[obs][len(test_data.collect()[0])-1])
    return acc / test_data.count(), preds, actuals"""