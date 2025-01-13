

from constants import *
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, BooleanType


### On this module, add functions, where each function represents one or more types of intrusions



## Gateway and device changes
def device_gateway_analysis(rssi, lsnr, tmst, len):


    # TODO: implement this  

    
    pass



# just for test
@udf(BooleanType())
def jamming_detection(rssi_values):
    
    if isinstance(rssi_values, list):
        for i in rssi_values:
            if (i is not None) and (i < RSSI_MIN or i > RSSI_MAX):
                return True

    return False

    