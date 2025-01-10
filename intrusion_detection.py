

from constants import *



### On this module, add functions, where each function represents one or more types of intrusions



## Gateway and device changes
def device_gateway_analysis(rssi, lsnr, tmst, len):


    # TODO: implement this  

    
    pass



# just for test
def jamming_detection(rssi_df):
    for element in rssi_df.collect():
        for i in element[0]:
            if i is not None and (i < RSSI_MIN or i > RSSI_MAX):
                return True

    return False

    