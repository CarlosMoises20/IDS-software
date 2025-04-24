
from pyspark.sql import SparkSession
from processing.message_classification import MessageClassification
from common.constants import *
from common.auxiliary_functions import create_spark_session


# TODO: implement a version that receives new messages in real time (stream processing), using the models ('results')
# retrieved from MLFlow



if __name__ == '__main__':

    spark_session = create_spark_session()

    # Initialize class used for network intrusion detection
    mc = MessageClassification(spark_session)

    # TODO: implement a script that classifies new messages in real time based on the corresponding models, probably in this way
    # mc.classify_new_incoming_messages()