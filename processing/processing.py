
from preProcessing.pre_processing import *
from auxiliaryFunctions.general_functions import *
from auxiliaryFunctions.ids_functions import *
from abc import ABC, abstractmethod


class DataProcessing(ABC):

    @abstractmethod
    def process_data(df_train, df_test, message_types):
        """..."""




