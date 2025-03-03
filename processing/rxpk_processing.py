
from processing.processing import DataProcessing
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from auxiliaryFunctions.general_functions import *


class RxpkProcessing(DataProcessing):
    
    @staticmethod
    def process_data(df_train, df_test):

        # TODO: continue training the model

        # Definir colunas categóricas com poucas categorias (que podemos indexar)
        limited_categorical_columns = ["MType", "NetID", "codr", "datr"]

        # Criar indexadores para essas colunas
        indexers = [StringIndexer(inputCol=col, outputCol=col + "_indexed", handleInvalid="keep") 
                    for col in limited_categorical_columns]

        # Definir colunas numéricas
        numerical_columns = ["DLSettingsRX1DRoffset", "DLSettingsRX2DataRate", 
                            "freq", "lsnr", "rssi", "size", "tmst"]

        # Criar lista final de colunas para o modelo
        feature_columns = numerical_columns + [col + "_indexed" for col in limited_categorical_columns]

        # Criar o vetor de features
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="keep")

        # Criar o modelo RandomForest
        rf = RandomForestClassifier(featuresCol="features", labelCol="intrusion", numTrees=100, maxDepth=10)

        # Criar pipeline
        pipeline = Pipeline(stages=indexers + [assembler, rf])

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