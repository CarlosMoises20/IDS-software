# IDS-Software
This repository contains the base software implementation for the Intrusion Detection System for LoRaWAN Networks, developed during the final project of MSc in Informatic and Computers Engineering, by the student Carlos Tavares (number 48725) of ISEL Lisbon School Engineering, in 2024/25 school year


### To install Python on your local machine
https://www.python.org/downloads/


### Install the necessary Python packages on your local machine
```
pip install -r ./requirements.txt
```


### To initialize CrateDB, you must first install Docker on your local machine
On Windows 10/11: https://docs.docker.com/desktop/setup/install/windows-install/

On Linux: https://docs.docker.com/desktop/setup/install/linux/ 

On Mac: https://docs.docker.com/desktop/setup/install/mac-install/  


### Then, run the following commands to create the CrateDB docker container where all data will be stored 
```
cd ./database
docker compose up
```


### Train and test the model, and store it on the database
```python3
python .\message_classification.py
```