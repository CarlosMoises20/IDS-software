
import argparse
from pyspark.sql.types import *
from common.auxiliary_functions import *
from prepareData.prepareData import prepare_past_dataset
from models.functions import *
from processing.message_classification import MessageClassification
from concurrent.futures import ProcessPoolExecutor
import time, os


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Message Classification for a specific DevAddr')
    parser.add_argument('--dev_addr', type=str, nargs='+', required=True, help='List of DevAddr to filter by')
    args = parser.parse_args()
    dev_addr_list = args.dev_addr
    
    # Initialize Spark Session
    spark_session = create_spark_session()
    
    #spark_session.sparkContext.setLogLevel("DEBUG")

    df = prepare_past_dataset(spark_session)

    # Splits the dataframe into "SPARK_NUM_PARTITIONS" partitions during pre-processing
    df = df.coalesce(numPartitions=int(SPARK_NUM_PARTITIONS))

    start_time = time.time()

    mc = MessageClassification(spark_session)

    # create all models in parallel to accelerate the code execution
    """df.groupBy("DevAddr").applyInPandas(
        lambda pdf: create_model_on_partition(pdf),
        schema=your_output_schema
    )"""

    # TODO: review this approach, it's not probably the best
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(mc.create_model, df, dev_addr)
                            for dev_addr in dev_addr_list]
        
        # Waits until all threads get completed
        for future in futures:
            future.result()


    end_time = time.time()

    # Print the total time of processing; the time is in seconds, minutes or hours
    print("Time of processing:", format_time(end_time - start_time), "\n\n")


    spark_session.stop()

