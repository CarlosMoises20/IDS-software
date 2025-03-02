from enum import Enum

"""
Class that defines all types of LoRaWAN datasets

"""
class DatasetType(Enum):
    RXPK = "rxpk"
    TXPK = "txpk"