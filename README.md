# IDS-Software
This repository contains the base software implementation for the Intrusion Detection System for LoRaWAN Networks, developed during the final project of MSc in Informatic and Computers Engineering, by the student Carlos Tavares (number 48725) of ISEL Lisbon School Engineering, in 2024/25 school year

### To install Apache Spark on your local machine (use version 3.3.2)
https://spark.apache.org/downloads.html

That will download a zip file. Then, you must follow the instructions on the site (such as verifying the integrity of the file) and choose a directory on your local machine to store the zip file.

You can also choose to use spark in a docker container if you want. To do so, first install Docker Desktop on your local machine, download the Spark docker image and then create a Docker container based on that image (see NOTE 1 in the end of the file)

### To install Java on your local machine (version 11), using the link below
https://www.oracle.com/java/technologies/javase/jdk11-archive-downloads.html

Or use the command line
&emsp; 3a - On Linux
```
sudo apt install openjdk-11-jdk
``` 

&emsp; 3b - On MacOS
```
brew apt install openjdk@11
``` 

&emsp; 3c - On Windows
```
choco install openjdk11
``` 

### Check Java version in your local machine to ensure that it was correctly installed
 ```
java --version
```

### To install Python on your local machine (version 3.10 recommended to avoid compatibility issues)
https://www.python.org/downloads/

### Check Python version in your local machine to ensure that it was correctly installed
 ```python3
python --version
```

### If pip is not automatically installed on your local machine after installing Python
Download this file here: https://bootstrap.pypa.io/get-pip.py (using curl on Windows or wget on Linux)

Then run the following command
```python
python ./get-pip.py
```

### Install the necessary Python packages on your local machine (in root directory)
```
pip install -r ./requirements.txt
```

#### NOTE: Properly define the environment variables on your local machine

### To install Git on your local machine
Go to the link https://git-scm.com/downloads

### To use Isolation Forest algorithm on the implementation, you must apply the following steps

1 - Clone the GitHub repository corresponding to the link https://github.com/linkedin/isolation-forest
```
git clone https://github.com/linkedin/isolation-forest
```
And then follow only the steps on the corresponding README file, adding that you need to properly modify the corresponding POM file if you want to use a different version for Java, Maven, Scala, Spark, etc. 

2 - Install Scala (version 2.12.12 recommended): https://www.scala-lang.org/download/ (or in the command line)

3 - Install sbt package

&emsp; 3a - On Linux
```
sudo apt install sbt
``` 

&emsp; 3b - On MacOS
```
brew apt install sbt
``` 

&emsp; 3c - On Windows
```
choco install sbt
``` 


### If you want to generate the input datasets in separate, with data pre-processing included run the following command (in root directory). You can also define in what format you want to generate the input datasets (JSON or PARQUET), and if you want to skip dataset generation if it already exists
```python3
python .\generate_input_datasets.py --dataset_format ("json" or "parquet"; by default is "parquet") --skip_dataset_generation_if_exists ("True" or "False"; by default is True)
```


### Create models based on specific devices (train and test) whose DevAddr can specified on the command line (example of DevAddr: "26012619"; don't forget the quotes), and save it as an MLFlow artifact (in root directory); if DevAddr's are not specified, all devices' address will be used as default
```python3
python .\create_models.py --dev_addr {DevAddr 1} {DevAddr 2} ... {DevAddr N} (by default, all devices' DevAddr) --dataset_format ("json" or "parquet"; by default is "parquet") --skip_dataset_generation_if_exists ("True" or "False"; by default is True)
``` 

### Run the IDS to receive and process new LoRaWAN messages in real time (stream processing) (in root directory)
 ```python3
python .\real_time_msg_processing.py --dataset_format ("json" or "parquet"; by default is "parquet") --skip_dataset_generation_if_exists ("True" or "False"; by default is True)
```
If you want to stop the application, just force it (Ctrl + C in Windows)


## NOTE 1: If you want to use Spark on Docker

### Install Docker Desktop

On Windows 10/11: https://docs.docker.com/desktop/setup/install/windows-install/

On Linux: https://docs.docker.com/desktop/setup/install/linux/

On Mac: https://docs.docker.com/desktop/setup/install/mac-install/


### Pull Spark Docker image and create a container

Link: https://hub.docker.com/_/spark

```
docker pull <name of image>
docker docker run -d --name spark-master \
  -h spark-master \
  -p 8085:8080 \
  -p 7077:7077 \
  <name of image> \
  spark-class org.apache.spark.deploy.master.Master
```

#### Check all existing docker images
```
docker images
```

#### Check all existing docker containers
```
docker ps -a
```

#### Check all running docker containers
```
docker ps
```


NOTE: to store in MLFLow the artifacts of all models, around 10GB - 12GB of free space will be needed in your machine