
import argparse
from pyspark.sql.types import *
from common.auxiliary_functions import *
from models.functions import *
from processing.message_classification import MessageClassification


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Message Classification for a specific DevAddr')
    parser.add_argument('--dev_addr', type=str, nargs='+', required=True, help='List of DevAddr to filter by')
    args = parser.parse_args()
    dev_addr_list = args.dev_addr
    
    # Initialize Spark Session
    spark_session = create_spark_session()
    
    # If you want to see spark errors in debug level on console during the script running
    #spark_session.sparkContext.setLogLevel("DEBUG")

    mc = MessageClassification(spark_session)

    mc.create_ml_models(dev_addr_list)

    spark_session.stop()

