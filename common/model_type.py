from enum import Enum

"""
Class that defines all model types used in the IDS

"""
class ModelType(Enum):
    LOF = {
        "name": "Local Outlier Factor", 
        "acronym": "lof", 
        "type": "sklearn"
    }
    IF_CUSTOM = {
        "name": "Isolation Forest (Custom)", 
        "acronym": "if_custom", 
        "type": "java_estimator"
    }
    IF_SKLEARN = {
        "name": "Isolation Forest (Sklearn)", 
        "acronym": "if_sklearn", 
        "type": "sklearn"
    }
    HBOS = {
        "name": "Histogram-Based Outlier Score", 
        "acronym": "hbos", 
        "type": "pyod"
    }
    KNN = {
        "name": "k-Nearest Neighbors", 
        "acronym": "knn", 
        "type": "spark"
    }
    OCSVM = {
        "name": "One-Class Support Vector Machine", 
        "acronym": "ocsvm", 
        "type": "sklearn"
    }