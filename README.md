# IDS-Software
This repository contains the base software implementation for the Intrusion Detection System for LoRaWAN Networks, developed during the final project of MSc in Informatic and Computers Engineering, by the student Carlos Tavares (number 48725) of ISEL Lisbon School Engineering, in 2024/25 school year


### Install the necessary packages on your local machine
```
pip install -r ./requirements.txt
```

### Run the project
```python3
python .\main.py
```

```utils.py```: This module contains functions to process each file considering the type of messages inside it

```attack_detection.py```: This module contains functions to detect each attack, using one or many ML algorithms for it


### Run some experiences to play with spark functionalities
```python3
python .\test.py
```
