
from processing.message_classification import MessageClassification
from common.constants import *
from common.auxiliary_functions import create_spark_session


if __name__ == '__main__':

    # Initialize Spark Session
    spark_session = create_spark_session()
    
    #spark_session.sparkContext.setLogLevel("DEBUG")

    # Initialize the class used for network intrusion detection
    mc = MessageClassification(spark_session)
    
    # call function to create ML models based on past data (batch processing)
    mc.create_ml_models()

    # Stop the Spark Session
    spark_session.stop()