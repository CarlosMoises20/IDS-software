
from functions import euclidean_dist



# define knn algorithm
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
    return max_class



# get accuracy of classifier
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
    return acc / test_data.count(), preds, actuals