# IDS-Software
This repository contains the base software implementation for the Intrusion Detection System for LoRaWAN Networks, developed during the final project of MSc in Informatic and Computers Engineering, by the student Carlos Tavares (number 48725) of ISEL Lisbon School Engineering, in 2024/25 school year

### To install Apache Spark on your local machine
https://spark.apache.org/downloads.html

That will download a zip file. Then, you must follow the instructions on the site (such as verifying the integrity of the file) and choose a directory on your local machine to store the zip file. 

### To install Python on your local machine (version 3.10 recommended due to compatibility issues)
https://www.python.org/downloads/

### Check the version of Python in your local machine
 ```python3
python --version
```

### If pip is not automatically installed on your local machine after installing Python
Download this file here: https://bootstrap.pypa.io/get-pip.py (using curl on Windows or wget on Linux)

Then run the following command
 ```python3
python ./get-pip.py
```


### Install the necessary Python packages on your local machine (in root directory)
```
pip install -r ./requirements.txt
```

#### NOTE: Properly define the environment variables on your local machine

### If you want to generate the input datasets in separate, run the following command (in root directory)
```python3
python .\generate_input_datasets.py
```


### Create the models (train and test), and save them as an MLFlow artifact (in root directory)
```python3
python .\create_models.py
```

### Create a specific model based on specific devices (train and test) whose DevAddr must be specified on the command line (example of DevAddr: "26012619"), and save it as an MLFlow artifact (in root directory)
```python3
python .\create_device_model.py --dev_addr {DevAddr 1} {DevAddr 2} ... {DevAddr N}
```

### Run the IDS to receive and process new LoRaWAN messages in real time (stream processing) (in root directory)
 ```python3
python .\real_time_msg_processing.py
```
If you want to stop the application, just force it (Ctrl + C in Windows)

