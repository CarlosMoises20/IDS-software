import base64

from constants import *
from crate.client import connect
import machine
import json

connection = connect(NODEURL)
cursor = connection.cursor()
cursor.execute(f"SELECT message, devaddr, tmst FROM {TABLE_SENSORS} where devaddr='583febc17ac93e82' ORDER BY tmst")
data = cursor.fetchall()
"""
devaddr = str(arg[1])           #devaddr
    tmst_actual = int(arg[2])       #tmst

    packet =[]     
    packet.append(float(arg[3]))    #latitude
    packet.append(float(arg[4]))    #longitude
    packet.append(int(arg[5]))      #sf
    packet.append(int(arg[6]))      #bw
    packet.append(float(arg[7]))    #lsnr
    packet.append(float(arg[8]))    #rssi
    packet.append(int(arg[9]))      #lenpayload
    packet.append(int(arg[12]))     #tmst_dif
"""
for (message, devaddr, tmst) in data:
    machine.main(['', devaddr, tmst, message['latitude'], message['longitude'], message['sf'], message['bw'], message['lsnr'], message['rssi'], message['lenpayload'], '', '', message['tmst_dif']])

cursor.close()