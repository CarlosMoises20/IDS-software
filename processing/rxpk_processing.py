
from processing.processing import DataProcessing
from pyspark.ml.feature import VectorAssembler
from auxiliaryFunctions.general_functions import *


class RxpkProcessing(DataProcessing):
    
    def process_data(df_train, df_test, message_types):

        # TODO: continue

        # Possible approach for numeric attributes
        #assembler = VectorAssembler(inputCols=get_numeric_attributes(df_train.schema), outputCol="features", handleInvalid="keep")

        # separate "rxpk" and "txpk" logic

        # TODO: study approach for categorical attributes or an approach for all types of attributes

        #rf = RandomForestClassifier(numTrees=5, maxDepth=4)

        #rf_model = rf.fit(df_train)

        #rf_predictions = rf_model.transform(df_test)

        # TODO: calculate "RFU", it comes from various attributes


        return 33