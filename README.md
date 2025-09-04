# IDS-Software
This repository contains the base software implementation for the Intrusion Detection System for LoRaWAN Networks, developed during the final project of MSc in Informatic and Computers Engineering, by the student Carlos Tavares (number 48725) of ISEL Lisbon School Engineering, in 2024/25 school year

### To install Apache Spark on your local machine (use version 3.3.2)
https://spark.apache.org/downloads.html

That will download a zip file. Then, you must follow the instructions on the site (such as verifying the integrity of the file) and choose a directory on your local machine to store the zip file.

You can also choose to install and use spark in a docker container if you want. To do so, first install Docker Desktop on your local machine, download the Spark docker image and then create a Docker container based on that image.

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

### To use the custom version of Isolation Forest algorithm on the implementation, you must apply the following steps

1 - Clone the GitHub repository corresponding to the link https://github.com/linkedin/isolation-forest
```
git clone https://github.com/linkedin/isolation-forest
```
And then follow only the steps on the corresponding README file, adding that you need to properly modify the corresponding POM file if you want to use a different version for Java, Maven, Scala, Spark, etc. Save the JAR file in the "jars" directory of your spark installation, and make sure SPARK_HOME environment variable is properly defined on your local machine

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


### To install Apache Kafka

You must install Scala first (version 2.12) as previously indicated. Then, ensure Java JDK 11 is properly installed on yout local machine, since its also a pre-requisite to install Apache Kafka (whether on Windows, Mac or Linux), and its installation instructions are above in this file.

Then go to the link https://kafka.apache.org/downloads and install the Kafka latest version on your local machine. Use the binary version and choose the latest version that has support for Scala 2.12, and then after downloading it, check its integrity by verifying if its ASC or SHA512 signature matches the one indicated on the site and unzip the file. One of the solutions to unzip the .tgz file is to execute the following command (example for Kafka version 3.9.1):

```
tar -xvzf kafka_2.12-3.9.1.tgz
```

Also make sure to properly define the environment variable to point it to the Kafka installation directory (KAFKA_HOME).

If you install Kafka on Windows, you will have to previously install WSL2 on your machine.

After installing Kafka, you will want to start Apache KRaft, which removes Kafka dependency from Zookeeper, simplifying Kafka architecture. To do so, whether on Windows (10 or above), Mac or Linux, you will have to perform the following steps:

&emsp; Move to the Kafka installation path
```
cd <kafka_path>
```

&emsp; Generate a new ID for your cluster running this command. The return of this command will be a generated UUID, which will be used in the next command (&lt;uuid&gt;) 
```
~/bin/kafka-storage.sh random-uuid
```

&emsp; Format your storage directory that is in the log.dirs in the config/kraft/server.properties, since the default directory is "/tmp/kraft-combined-logs".
```
~/bin/kafka-storage.sh format -t <uuid> -c ~/config/kraft/server.properties
```

&emsp; Finally, you can launch the broker itself in daemon mode by running this command. This is necessary to Spark to connect with Kafka to consume the LoRa messages coming from the UDP socket and stored on the created Kafka topic.
```
~/bin/kafka-server-start.sh ~/config/kraft/server.properties
```

&emsp; To stop the application, use Ctrl + C to force it. But when you start it again, it's a good practice to previously stop any running Kafka process, executing the command below.
```
~/bin/kafka-server-stop.sh ~/config/kraft/server.properties
```

These instructions to install Apache KRaft are also available on these sites below:

&emsp; On Windows 10 or above
```
https://learn.conduktor.io/kafka/how-to-install-apache-kafka-on-windows-without-zookeeper-kraft-mode/
```

&emsp; On Mac
```
https://learn.conduktor.io/kafka/how-to-install-apache-kafka-on-mac-without-zookeeper-kraft-mode/
```

&emsp; On Linux
```
https://learn.conduktor.io/kafka/how-to-install-apache-kafka-on-linux-without-zookeeper-kraft-mode/
```


You also have to install the kafka connector for Spark. To do so, go to https://mvnrepository.com/artifact/org.apache.spark/spark-sql-kafka-0-10_2.12/3.3.2 and download the JAR by clicking on "jar" in Files section. Then, move the JAR file to the "jars" subdirectory in your Spark installation, making sure that SPARK_HOME is properly defined in your local machine.

Then to the same thing for these JARs:

https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/2.8.0/kafka-clients-2.8.0.jar

https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.3.2/spark-token-provider-kafka-0-10_2.12-3.3.2.jar

https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.12.1/commons-pool2-2.12.1.jar


### If you want to generate the input datasets in separate, with data pre-processing included run the following command (in root directory). You can also define in what format you want to generate the input datasets (JSON or PARQUET), and if you want to skip dataset generation if it already exists. PARQUET allows a faster processing but JSON allows you to see the content in a legible format.
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



