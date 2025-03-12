
from processing.processing import DataProcessing
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Model
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from auxiliaryFunctions.general import get_all_attributes_names


class RxpkProcessing(DataProcessing):
    
    @staticmethod
    def process_data(df_train, df_test):

        # TODO: continue training the model

        # get all attributes names to assemble since they are all now numeric
        column_names = get_all_attributes_names(df_train.schema)

        # Create the VectorAssembler that merges all features of the dataset into a Vector
        # These feature are, now, all numeric and with the missing values all imputed, so now we can use them
        assembler = VectorAssembler(inputCols=column_names, outputCol="features")

        # Create the Random Forest Classifier model
        rf = RandomForestClassifier(featuresCol="features", labelCol="intrusion", numTrees=7, maxDepth=5, seed=522)
 
        pipeline = Pipeline(stages=[assembler, rf])

        # Treinar o modelo
        model = pipeline.fit(df_train)

        # Fazer previsões
        predictions = model.transform(df_test)

        # Avaliar o modelo
        evaluator = BinaryClassificationEvaluator(labelCol="intrusion", metricName="areaUnderROC")
        roc_auc = evaluator.evaluate(predictions)

        print(f"Área sob a curva ROC: {roc_auc}")


        return 3