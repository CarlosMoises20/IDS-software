# IDS-Software
This repository contains the base software implementation for the Intrusion Detection System for LoRaWAN Networks, developed during the final project of MSc in Informatic and Computers Engineering, by the student Carlos Tavares (number 48725) of ISEL Lisbon School Engineering, in 2024/25 school year

### To install Apache Spark on your local machine
https://spark.apache.org/downloads.html

That will download a zip file. Then, you must follow the instructions on the site (such as verifying the integrity of the file) and choose a directory on your local machine to store the zip file. 

### To install Python on your local machine (version 3.10 recommended due to compatibility issues)
https://www.python.org/downloads/


### Install the necessary Python packages on your local machine (in root directory)
```
pip install -r ./requirements.txt
```

#### NOTE: Properly define the environment variables on your local machine


### To initialize CrateDB, you must first install Docker on your local machine
On Windows 10/11: https://docs.docker.com/desktop/setup/install/windows-install/

On Linux: https://docs.docker.com/desktop/setup/install/linux/ 

On Mac: https://docs.docker.com/desktop/setup/install/mac-install/  


### Then, run the following commands to create the CrateDB docker container where all data will be stored 
```
cd ./database
docker compose up
```

### To create tables in the database, run the corresponding script with the following commands
```
cd ./database 
python ./init_db.py
```

### Train and test the models, and store them on the database (in root directory)
```python3
python .\message_classification.py
```

### Run the IDS to receive and process new LoRaWAN messages in real time (stream processing) (in root directory)
 ```python3
python .\real_time_msg_processing.py
```
If you want to stop the application, just force it (Ctrl + C in Windows)


### If you want to clean all data in the database (without removing the tables), run the corresponding script with the following commands
```
cd ./database 
python ./clean_db.py
```

### If you want to remove the container from your local machine, run the following commands
```
cd ./database
docker compose down
```

