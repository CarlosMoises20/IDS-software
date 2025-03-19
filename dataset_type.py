from enum import Enum
from preProcessing.rxpk_pre_processing import RxpkPreProcessing
from preProcessing.txpk_pre_processing import TxpkPreProcessing

"""
Class that defines all types of LoRaWAN datasets

"""
class DatasetType(Enum):
    RXPK = {
        "name": "rxpk", 
        "pre_processing_class": RxpkPreProcessing()
    }
    TXPK = {
        "name": "txpk", 
        "pre_processing_class": TxpkPreProcessing()
    }