from enum import Enum
from preProcessing.rxpk_pre_processing import RxpkPreProcessing
from processing.rxpk_processing import RxpkProcessing
from preProcessing.txpk_pre_processing import TxpkPreProcessing
from processing.txpk_processing import TxpkProcessing

"""
Class that defines all types of LoRaWAN datasets

"""
class DatasetType(Enum):
    RXPK = {
        "name": "RXPK",
        "filename_field": "rxpk", 
        "pre_processing_class": RxpkPreProcessing(),
        "processing_class": RxpkProcessing()
    }
    TXPK = {
        "name": "TXPK",
        "filename_field": "txpk", 
        "pre_processing_class": TxpkPreProcessing(),
        "processing_class": TxpkProcessing()
    }