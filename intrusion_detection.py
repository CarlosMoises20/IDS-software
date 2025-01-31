

from constants import *
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType


### On this module, add functions, where each function represents one or more types of intrusions




# TODO: complete this function
@udf(BooleanType())
def jamming_detection(rssi_values):
    
    if isinstance(rssi_values, list):
        for i in rssi_values:
            if (i is not None) and (i < RSSI_MIN or i > RSSI_MAX):
                return True

    return False

    