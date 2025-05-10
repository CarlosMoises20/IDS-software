
from enum import Enum

"""
Class that defines all types of LoRaWAN messages in datasets

"""
class DatasetFormatType(Enum):
    JSON = "json"
    PARQUET = "parquet"