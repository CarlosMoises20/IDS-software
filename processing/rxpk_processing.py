
from processing.processing import DataProcessing



class RxpkProcessing(DataProcessing):
    
    def __init__(self, spark_session, dataset, dataset_type):
        super().__init__(spark_session, dataset, dataset_type)

    def process_dataset(self):
        pass