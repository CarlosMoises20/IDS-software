
from processing.processing import DataProcessing
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.tree import RandomForest
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from auxiliaryFunctions.general_functions import *


class RxpkProcessing(DataProcessing):
    
    @staticmethod
    def process_data(df_train, df_test):

        # TODO: continue training the model

        feature_columns = get_all_attributes_names(df_train.schema)

        # Criar o vetor de features
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="keep")

        # Criar o modelo RandomForest
        rf = RandomForestClassifier(featuresCol="features", labelCol="intrusion", numTrees=100, maxDepth=10)

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


        """
        numeric_attributes = get_numeric_attributes(df_train.schema)

        # Possible approach for numeric attributes
        assembler = VectorAssembler(inputCols=numeric_attributes, outputCol="features", handleInvalid="keep")

        # Define Random Forest classifier (handles categorical & missing values)
        rf = RandomForestClassifier(featuresCol="features", numTrees=20)

        # Build pipeline
        pipeline = Pipeline(stages=[assembler, rf])

        # Train the model   
        model = pipeline.fit(df_train)

        # Evaluate the model
        predictions = model.transform(df_test.select(numeric_attributes))

        predictions.select("prediction", "probability").show(10)
        """


        return 3