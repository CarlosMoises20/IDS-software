
from pyspark.sql.types import *
from pyspark.sql.functions import expr, struct, col, explode
from abc import ABC, abstractmethod


class PreProcessing(ABC):

    def __init__(self, df):
        self.__df = df


    @abstractmethod
    def pre_process_dataset(self):
        pass

