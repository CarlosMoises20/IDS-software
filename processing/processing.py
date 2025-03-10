
from preProcessing.pre_processing import *
from auxiliaryFunctions.general import *
from auxiliaryFunctions.anomaly_detection import *
from abc import ABC, abstractmethod


class DataProcessing(ABC):

    @staticmethod
    @abstractmethod
    def process_data(df_train, df_test):
        """..."""