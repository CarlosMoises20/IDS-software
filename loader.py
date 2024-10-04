import base64

from constants import *
from crate.client import connect
import machine
import json

data1_file = open("dataset_json/ferrovia40_reports_0_.json", "r")
data2_file = open("dataset_json/ferrovia40_reports_1_.json", "r")
data3_file = open("dataset_json/ferrovia40_reports_2_.json", "r")
data4_file = open("dataset_json/ferrovia40_reports_3_.json", "r")

data_all_list = []
data_all_list.extend(data1_file.read().split("\n"))
data_all_list.extend(data2_file.read().split("\n"))
data_all_list.extend(data3_file.read().split("\n"))
data_all_list.extend(data4_file.read().split("\n"))
sensors_tmst_last_msg = {'583febc17ac93e82': 0, '27199a99ba2e0c14': 0, '5435ec9e93616f51': 0, '7c6d83a30e7b7fd1': 0}
data_all = list(filter(lambda elem: len(elem) > 0, data_all_list))
data_all = list(map(lambda x: json.loads(x), data_all))
data_all.sort(key=lambda x: x['time'])

connection = connect(NODEURL)
cursor = connection.cursor()
cursor.execute("DELETE FROM sensors")

def dr_to_sf(dr):
    if dr == 0:
        return 12
    elif dr == 1:
        return 11
    elif dr == 2:
        return 10
    elif dr == 3:
        return 9
    elif dr == 4:
        return 8
    elif dr == 5:
        return 7
    elif dr == 6:
        return 7


i = 0
for data in data_all:
    gatewayExists = False
    if 'gateways' in data:
        for gateway in data['gateways']:
            if str(gateway['name']).lower() == 'gateway2':
                gatewayExists = True
        if data['gateways'][0]['location']['latitude'] != 0 and 'dump' in data and 'txInfo' in data['dump'] and (
                data['time'] - sensors_tmst_last_msg[data['dev_eui']]) > 0 and gatewayExists:
            payload = base64.b64decode(data['dump']['data']).hex()
            message = {
                'devaddr': '583febc17ac93e82',
                'tmst': data['time'],
                'message': {
                    "bw": 0,
                    "flag": 0,
                    "latitude": data['gateways'][0]['location']['latitude'],
                    "longitude": data['gateways'][0]['location']['longitude'],
                    "lsnr": data['gateways'][0]['loRaSNR'],
                    "rssi": data['gateways'][0]['rssi'],
                    "tmst": data['time'],
                    "tmst_dif": data['time'] - sensors_tmst_last_msg[data['dev_eui']],
                    "lenpayload": len(payload),
                    "payload": payload,
                    "sf": dr_to_sf(data['dump']['txInfo']['dr'])
                }
            }
            i += 1
            sensors_tmst_last_msg[data['dev_eui']] = data['time']
            cursor.execute("INSERT INTO sensors (devaddr, tmst, message) VALUES (?, ?, ?)", (message['devaddr'], message['tmst'], message['message']))
            
print(i)        # 'i' represents the number of messages processed and inserted into the sensors table, effectively sending the data to the sensors
cursor.close()
