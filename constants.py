
## All constants are defined here


'''CRATEDB'''
HOST = 'localhost'
PORT = 4201
DATABASE_URI = f'http://{HOST}:{PORT}'


'''ANOMALY DETECTION'''
LSNR_MIN = -20
LSNR_MAX = 10
RSSI_MIN = -130
RSSI_MAX = -10
TMST_MIN = 2000     # miliseconds
LEN_MIN = 5        # REVIEW
LEN_MAX = 20       # REVIEW


# Expected frequency values (LoRaWAN operates on specific frequencies)
EXPECTED_FREQUENCIES = [868.1, 868.3, 868.5, 868.7, 868.9]  # EU868 example

# Maximum allowed timestamp difference (adjust based on real data)
EXPECTED_TMST_THRESHOLD = 1000000  # Example threshold (microseconds)
