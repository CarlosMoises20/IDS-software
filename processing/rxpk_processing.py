
import time
from processing.processing import DataProcessing
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Model
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from auxiliaryFunctions.general import get_all_attributes_names, format_time


class RxpkProcessing(DataProcessing):
    
    @staticmethod
    def process_data(df):

        start_time = time.time()

        # get all attributes names to assemble since they are all now numeric
        column_names = get_all_attributes_names(df.schema)

        # Create the VectorAssembler that merges all features of the dataset into a Vector
        # These feature are, now, all numeric and with the missing values all imputed, so now we can use them
        assembler = VectorAssembler(inputCols=column_names, outputCol="features")
 
        pipeline = Pipeline(stages=[assembler])

        # Train the pipeline model to assemble all features
        pipelineModel = pipeline.fit(df)
        df = pipelineModel.transform(df)

        # randomly divide dataset into training (80%) and test (20%)
        # and set a seed in order to ensure reproducibility, which is important to 
        # ensure that the model is always trained and tested on the same examples each time the
        # model is run. This is important to compare the model's performance in different situations
        # (this proportion can be modified according to the results)
        df_train, df_test = df.randomSplit([0.8, 0.2], seed=522)

        # Create the Random Forest Classifier model
        rf = RandomForestClassifier(featuresCol="features", labelCol="intrusion", 
                                    numTrees=6, maxDepth=3, seed=522, maxMemoryInMB=8192)

        model = rf.fit(df_train)

        # Fazer previsões
        predictions = model.transform(df_test)

        # Avaliar o modelo
        evaluator = BinaryClassificationEvaluator(labelCol="intrusion", metricName="areaUnderROC")
        roc_auc = evaluator.evaluate(predictions)

        print(f"Accuracy: {roc_auc}")

        end_time = time.time()

        print("Time of rxpk processing: ", format_time(end_time - start_time), "\n\n")

        # TODO: the model and evaluator should be returned to be used on real-time incoming messages
        return 3