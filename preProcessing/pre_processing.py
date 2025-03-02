
from pyspark.sql.types import *
from abc import ABC, abstractmethod


class DataPreProcessing(ABC):

    @abstractmethod
    def pre_process_data(df):
        """..."""

