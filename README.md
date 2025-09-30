# IDS-Software
This repository contains the base software implementation for the Intrusion Detection System for LoRaWAN Networks, developed during the final project of MSc in Informatic and Computers Engineering, by the student Carlos Tavares (number 48725) of ISEL Lisbon School Engineering, in 2024/25 school year


Firstly, for Windows users, it's recommended to use WSL, executing the following command to install a Linux environment (for example, Ubuntu):

```
wsl --install
```

After WSL being installed, you will define your username and password. This allows you to have an isolated environment where you can install all necessary packages to run this solution, avoiding possible conflicts with your local machine.

### To install Apache Hadoop (version 3 recommended, for example 3.4.0)

https://hadoop.apache.org/releases.html

Go to "Downloads" and click on the link on HTTP. Then, verify the integrity of the tgz file, extract the file and store the extracted folder on any directory. Finally, set HADOOP_HOME according to the directory you have chosen, and add the bin folder to PATH. You can also download the installer instead. 


### To install Apache Spark (version 3.3.2 recommended)
https://spark.apache.org/downloads.html (go to the "Archived releases" section or similar to find a link with older Spark releases which includes the version 3.3.2)

You must download the installer or download a tgz file which finishes with (bin-hadoop3.tgz), indicating that the Spark version uses the version 3 of Apache Hadoop. Then, you must follow the instructions on the site (such as verifying the integrity of the file) and choose a directory to store the tgz file. It's recommended to always check its integrity by verifying its signature. Don't forget to properly set the environment variable SPARK_HOME and add its bin folder to PATH.

### To install Java (version 11), use the link below
https://www.oracle.com/java/technologies/javase/jdk11-archive-downloads.html

Or use the command line

&emsp; 3a - On Linux
```
sudo apt install openjdk-11-jdk -y
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

### To install Python (version 3.10 recommended to avoid compatibility issues)
https://www.python.org/downloads/

### Check Python version in your local machine to ensure that it was correctly installed
 ```python3
python --version
```

### If pip is not automatically installed after installing Python
Download this file here: https://bootstrap.pypa.io/get-pip.py

Then run the following command
```python
python ./get-pip.py
```

Finally, ensure that the PATH environment variable contains the folder where the "pip" file is located. 

### Install the necessary Python packages (and all their dependencies) (in root directory)
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

Install python virtual environment with this command (you can do this in an isolated environment to avoid conflicts with your current Python installation):

```
sudo apt install python3.10-venv
```

And then follow only the steps on the corresponding README file, adding that you need to properly modify the corresponding POM file if you want to use a different version for Java, Maven, Scala, Spark, etc. Save the JAR file in the "jars" directory of your spark installation, and make sure SPARK_HOME environment variable is properly defined on your local machine, as well as Scala, Java, Maven and Python environment variables.

2 - Install Scala (version 2.12.12 recommended): https://www.scala-lang.org/download/ (or in the command line)

3 - Install sbt package

&emsp; 3a - On Linux
```
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update
sudo apt-get install sbt
``` 

&emsp; 3b - On MacOS
```
brew apt install sbt
``` 

&emsp; 3c - On Windows
```
choco install sbt
``` 

Then, if you installed this on the same machine than your other Python installation, to avoid conflicts, you can uninstall this python virtual environment.

### To install Apache Kafka

You must install Scala first (version 2.12) as previously indicated. Then, ensure Java JDK 11 is properly installed on your local machine, since it's also required to install Apache Kafka (whether on Windows, Mac or Linux), and its installation instructions are above in this file.

Then go to the link https://kafka.apache.org/downloads and install the Kafka latest version on your local machine. Use the binary version and choose the latest version that has support for Scala 2.12, and then after downloading it, check its integrity by verifying if its ASC or SHA512 signature matches the one indicated on the site and unzip the file. One of the solutions to unzip the .tgz file is to execute the following command (example for Kafka version 3.9.1). For example, on Linux, you can run this command below:

```
tar -xzf kafka_2.12-3.9.1.tgz
```

Also make sure to properly define the environment variable to point it to the Kafka installation directory (KAFKA_HOME).

Remember that if you install Kafka on Windows, you will have to previously install WSL2 on your machine.

After installing Kafka, you will want to start Apache KRaft, which removes Kafka dependency from Zookeeper, simplifying Kafka architecture. To do so, whether on Windows (10 or above), Mac or Linux, you will have to perform the following steps:

&emsp; Move to the Kafka installation path
```
cd <kafka_path>
```

&emsp; Generate a new ID for your cluster running this command. The return of this command will be a generated UUID, which will be used in the next command (&lt;uuid&gt;) 
```
bin/kafka-storage.sh random-uuid
```

&emsp; Format your storage directory that is in the log.dirs in the config/kraft/server.properties, since the default directory is "/tmp/kraft-combined-logs".
```
bin/kafka-storage.sh format -t <uuid> -c config/kraft/server.properties
```

&emsp; Finally, you can launch the broker itself in daemon mode by running this command. This is necessary to Spark to connect with Kafka to consume the LoRa messages coming from the UDP socket and stored on the created Kafka topic.
```
bin/kafka-server-start.sh config/kraft/server.properties
```

&emsp; To stop the application, use Ctrl + C to force it. But when you start it again, it's a good practice to previously stop any running Kafka process, executing the command below.
```
bin/kafka-server-stop.sh config/kraft/server.properties
```

These instructions to install Apache KRaft are also available on these links below:

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


# Additional necessary enviromnent variables

To run Spark jobs with PySpark, you need to set the environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON so they point to the Python interpreter you want to use (for example, python3).

In addition, you should set PYTHONPATH to include the python subdirectory of SPARK_HOME as well as the py4j-(version).zip file inside SPARK_HOME/python/lib. This ensures that the Spark workers can properly use Py4J to connect Python with the JVM, avoiding related errors.

### If you want to generate the input datasets in separate

It includes data pre-processing included run the command below.

--dataset_format: in what format you want to generate the input datasets (JSON or PARQUET); PARQUET allows a faster processing but JSON allows you to see the content in a legible format.

--skip_dataset_generation_if_exists: if you want to skip dataset generation if it already exists (True) or not (False)

```python3
python .\generate_input_datasets.py --dataset_format ("json" or "parquet"; by default is "parquet") --skip_dataset_generation_if_exists ("True" or "False"; by default is True)
```


### Create models based on specific devices (train and test) whose DevAddr can specified on the command line, and save it as an MLFlow artifact (in root directory)

--dev_addr: DevAddr of devices from which you want to create models (example of DevAddr: "26012619"; don't forget the quotes); if DevAddr's are not specified, all devices' address will be used as default

--dataset_format: in what format you want to generate the input datasets (JSON or PARQUET); PARQUET allows a faster processing but JSON allows you to see the content in a legible format.

--skip_dataset_generation_if_exists: if you want to skip dataset generation if it already exists (True) or not (False)

--with_feature_scaling: if you want to apply feature scaling on data pre-processing (True) or not (False)

--with_feature_reduction: if you want to apply feature reduction or not; if you want, you can choose between PCA and SVD. Otherwise, type "None"

```python3
python .\create_models.py --dev_addr {DevAddr 1} {DevAddr 2} ... {DevAddr N} (by default, all devices' DevAddr) --dataset_format ("json" or "parquet"; by default is "parquet") --skip_dataset_generation_if_exists ("True" or "False"; by default is True) --with_feature_scaling ("True" or "False"; by default "True") --with_feature_reduction ("PCA", "SVD" or "None"; by default "SVD")
``` 

### Run the IDS to receive and process new LoRaWAN messages in real time (stream processing) (in root directory)

 ```python3
python .\real_time_msg_processing.py --dataset_format ("json" or "parquet"; by default is "parquet") --skip_dataset_generation_if_exists ("True" or "False"; by default is True) --with_feature_scaling ("True" or "False"; by default "True") --with_feature_reduction ("PCA", "SVD" or "None"; by default "SVD")
```
