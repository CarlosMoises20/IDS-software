from constants import *
from crate.client import connect

connection = connect(NODEURL)
cursor = connection.cursor()
cursor.execute(f"SELECT message, devaddr, tmst FROM {TABLE_SENSORS} where devaddr='583febc17ac93e82' ORDER BY tmst")
data = cursor.fetchall()
cursor.close()
header = "id;lenpayload;lsnr;rssi;latitude;longitude;flag"
file = header + "\n"
id = 1
for (message, devaddr, tmst) in data:
    file += str(id) + ";" + str(message['lenpayload']) + ";" + str(message['lsnr']) + ";" + str(
        message['rssi']) + ";" + str(message['latitude']).replace(".",",") + ";" + str(message['longitude']).replace(".",",") + ";" + str(
        message['flag']) + "\n"
    print("%2d: %s" % (message['flag'], message))
    id += 1

stats = open("dataset_json/results_pos_detection.csv", "w")
stats.write(file)
stats.close()
