
import time
from processing.processing import DataProcessing
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from auxiliaryFunctions.general import get_all_attributes_names, format_time


class TxpkProcessing(DataProcessing):

    @staticmethod
    def process_data(df):

        start_time = time.time()

        # TODO: continue training the model

        column_names = get_all_attributes_names(df.schema)

        # merge all features into a Vector
        assembler = VectorAssembler(inputCols=column_names, outputCol="features")

        # Create pipeline
        pipeline = Pipeline(stages=[assembler])

        # Apply the pipeline to the dataset to assemble the features
        pipelineModel = pipeline.fit(df)
        df = pipelineModel.transform(df)

        # randomly divide dataset into training (80%) and test (20%)
        # and set a seed in order to ensure reproducibility, which is important to 
        # ensure that the model is always trained and tested on the same examples each time the
        # model is run. This is important to compare the model's performance in different situations
        # (this proportion can be modified according to the results)
        df_train, df_test = df.randomSplit([0.8, 0.2], seed=522)

        # Criar o modelo RandomForest
        rf = RandomForestClassifier(featuresCol="features", labelCol="intrusion", numTrees=7, maxDepth=5, seed=522)

        model = rf.fit(df_train)

        # Calculate model predicions based on the learnt patterns during training
        predictions = model.transform(df_test)

        # Avaliar o modelo
        evaluator = BinaryClassificationEvaluator(labelCol="intrusion", metricName="areaUnderROC")
        accuracy = evaluator.evaluate(predictions)

        print(f"Accuracy: {accuracy}")

        end_time = time.time()

        print("Time of txpk processing:", format_time(end_time - start_time), "\n\n")

        # TODO: the model and evaluator should be returned to be used on real-time incoming messages
        return 1