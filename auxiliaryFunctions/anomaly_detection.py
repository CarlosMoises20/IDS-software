

from constants import *
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType
from scipy.spatial.distance import hamming
import numpy as np
from scipy.stats import entropy



class AnomalyDetection:

    """
    This function computes the hamming distance between two given strings
    that must have the same length

    The function returns None if at least one of the strings is None or 
    if the strings have different lengths

    """
    @staticmethod
    def __hamming_distance(str1, str2):
        
        if (str1 is None) or (str2 is None):
            return None
        
        if len(str1) != len(str2):
            return None
        
        return hamming(list(str1), list(str2))


    """
    This function computes the Kullback-Leibler Divergence between two given probability distributions

    P is the baseline probability distribution, the learnt distribution
    Q is the new probability distribution, the real-time distribution of the random number generated in the 
        Join-Request

    """
    @staticmethod
    def __kl_divergence(P, Q):

        P = np.array(P, dtype=np.float64)
        Q = np.array(Q, dtype=np.float64)

        # Avoid division by zero, replacing zero elements with a very small value (??)
        P[P == 0] = 1e-10
        Q[Q == 0] = 1e-10
        
        return entropy(P, Q, base=2)  # Using log base 2 for better interpretability



    # auxiliary boolean function to detect jamming attacks
    @staticmethod
    def __jamming_detection(rssi):
        return rssi < RSSI_MIN or rssi > RSSI_MAX
    

    # Replay attack detection based on FCnt reuse
    @staticmethod
    def __replay_attack(fcnt_history, current_fcnt):
        return current_fcnt in fcnt_history

    # Sinkhole attack detection based on frequency anomalies
    @staticmethod
    def __sinkhole_detection(freq):
        return freq not in EXPECTED_FREQUENCIES

    # Wormhole attack detection based on timestamp anomalies
    @staticmethod
    def __wormhole_detection(tmst):
        return tmst > EXPECTED_TMST_THRESHOLD

    # Downlink routing attack detection using MAC Payload validation
    @staticmethod
    def __downlink_routing_attack(valid_mac_payload):
        return valid_mac_payload == 0  # Invalid payload suggests interference

    # Physical tampering detection based on unexpected FHDR values
    @staticmethod
    def __physical_tampering(valid_fhdr):
        return valid_fhdr == 0
    

    @staticmethod
    def __detection(message_type):

        if message_type == 0:   # Join Request
            pass
        if message_type == 1:   # Join Accept
            pass
        if message_type == 2:   # Unconfirmed Data Up
            pass
        if message_type == 3:   # Unconfirmed Data Down
            pass
        if message_type == 4:   # Confirmed Data Up
            pass
        if message_type == 5:   # Confirmed Data Down
            pass
        if message_type == 7:   # Proprietary
            pass

        # Rejoin-Requests (6) don't exist on the dataset, so the model won't be trained on it



        

    