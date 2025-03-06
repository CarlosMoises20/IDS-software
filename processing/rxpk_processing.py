
from processing.processing import DataProcessing
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.tree import RandomForest
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from auxiliaryFunctions.general import *


class RxpkProcessing(DataProcessing):
    
    @staticmethod
    def process_data(df_train, df_test):

        # TODO: continue training the model

        column_names = get_all_attributes_names(df_train.schema)

        # Criar o vetor de features
        assembler = VectorAssembler(inputCols=column_names, outputCol="features", handleInvalid="keep")

        # Criar o modelo RandomForest
        rf = RandomForestClassifier(featuresCol="features", labelCol="intrusion", numTrees=7, maxDepth=5, seed=522)

        # Criar pipeline
        pipeline = Pipeline(stages=[assembler, rf])

        # Treinar o modelo
        model = pipeline.fit(df_train)

        # Fazer previsões
        predictions = model.transform(df_test)

        # Avaliar o modelo
        evaluator = BinaryClassificationEvaluator(labelCol="intrusion", metricName="areaUnderROC")
        roc_auc = evaluator.evaluate(predictions)

        print(f"Área sob a curva ROC: {roc_auc}")

        # Exibir previsões
        predictions.select("intrusion", "prediction", "probability").show(10)


        return 3