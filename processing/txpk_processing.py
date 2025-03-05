

from processing.processing import DataProcessing
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.mllib.tree import RandomForest
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


class TxpkProcessing(DataProcessing):

    @staticmethod
    def process_data(df_train, df_test):

        # TODO: continue training the model

        # Definir colunas numéricas
        feature_columns = ["DLSettingsRX1DRoffset", "DLSettingsRX2DataRate", 
                            "freq", "size", "tmst"]

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

        return 1