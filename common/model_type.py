from enum import Enum

"""
Class that defines all model types used in the IDS

"""
class ModelType(Enum):
    LOF = {
        "name": "Local Outlier Factor", 
        "type": "sklearn"
    }
    IF_CUSTOM = {
        "name": "Isolation Forest (Custom)", 
        "type": "java_estimator"
    }
    IF_SKLEARN = {
        "name": "Isolation Forest (Sklearn)", 
        "type": "sklearn"
    }
    HBOS = {
        "name": "Histogram-Based Outlier Score", 
        "type": "pyod"
    }
    KNN = {
        "name": "k-Nearest Neighbors", 
        "type": "spark"
    }
    OCSVM = {
        "name": "One-Class Support Vector Machine", 
        "type": "sklearn"
    }