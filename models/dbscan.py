

# inspiration: https://github.com/SalilJain/pyspark_dbscan/blob/master/dbscan.py


# TODO: fix or remove if not inside expectations

from pyspark.sql import Row
from graphframes import GraphFrame
from itertools import combinations
import math

class SparkDBSCAN:
    def __init__(self, epsilon, min_pts, dist, checkpoint_dir, operations=None):
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.dist = dist
        self.checkpoint_dir = checkpoint_dir
        self.operations = operations
        self.model = None  # vai guardar o resultado do cluster
    
    # Método para treinar / rodar o clustering e guardar o modelo
    def train(self, df):
        """
        df: DataFrame do spark com colunas: id, value (value é vetor ou ponto)
        """
        zero = df.rdd.takeSample(False, 1)[0].value
        combine_cluster_rdd = df.rdd.\
            flatMap(self.__distance_from_pivot(zero)). \
            reduceByKey(lambda x, y: x + y).\
            flatMap(self.__scan()). \
            reduceByKey(lambda x, y: x.union(y)).\
            flatMap(self.__label()).\
            reduceByKey(lambda x, y: x + y).map(self.__combine_labels).cache()

        self.model = combine_cluster_rdd
        return self.model

    # Método para fazer previsões com o modelo treinado
    def predict(self, df):
        if self.model is None:
            raise Exception("Modelo não treinado. Execute train() primeiro.")
        
        spark = df.sql_ctx.sparkSession
        id_cluster_rdd = self.model.\
            map(lambda x: Row(point=x[0], cluster_label=x[1][0], core_point=x[2]))

        id_cluster_df = id_cluster_rdd.toDF()
        vertices = self.model.\
            flatMap(lambda x: [Row(id=item) for item in x[1]]).toDF().distinct()
        edges = self.model. \
            flatMap(lambda x: [Row(src=item[0], dst=item[1])
                               for item in combinations(x[1], 2)]). \
            toDF().distinct()
        spark.sparkContext.setCheckpointDir(self.checkpoint_dir)
        g = GraphFrame(vertices, edges)
        connected_df = g.connectedComponents()
        result_df = id_cluster_df.\
            join(connected_df, connected_df.id == id_cluster_df.cluster_label). \
            select("point", "component", "core_point")

        # Juntar a predição ao DataFrame original
        prediction_df = df.join(result_df, df.id == result_df.point).select(df["*"], "component", "core_point")

        return prediction_df

    # Avaliar o resultado, exemplo simples com accuracy e matriz de confusão
    # (Para clustering isso é complexo, normalmente usamos métricas como silhouette, etc)
    def evaluate(self, prediction_df, true_labels_col):
        from pyspark.ml.evaluation import ClusteringEvaluator
        evaluator = ClusteringEvaluator(featuresCol="value", predictionCol="component")
        silhouette = evaluator.evaluate(prediction_df)
        print(f"Silhouette Score: {silhouette}")

        # Se tiver rótulos verdadeiros, pode criar matriz confusão simplificada
        # Aqui só um exemplo simples para ilustrar (você pode implementar conforme seus dados)
        prediction_df.groupBy(true_labels_col, "component").count().show()

        return silhouette
    
    # As funções internas convertidas para métodos para usar self
    def __distance_from_pivot(self, pivot):
        def distance(x):
            pivot_dist = self.dist(x.value, pivot)
            if self.operations is not None:
                self.operations.add()
            partition_index = math.floor(pivot_dist / self.epsilon)
            rows = [Row(id=x.id, value=x.value, pivot_dist=self.dist(x.value, pivot))]
            out = [(partition_index, rows),
                   (partition_index + 1, rows)]
            return out
        return distance

    def __scan(self):
        def scan(x):
            out = {}
            partition_data = x[1]
            partition_len = len(partition_data)
            for i in range(partition_len):
                for j in range(i + 1, partition_len):
                    if self.operations is not None:
                        self.operations.add()
                    if self.dist(partition_data[i].value, partition_data[j].value) < self.epsilon:
                        if partition_data[i].id in out:
                            out[partition_data[i].id].add(partition_data[j].id)
                        else:
                            out[partition_data[i].id] = set([partition_data[j].id])
                        if partition_data[j].id in out:
                            out[partition_data[j].id].add(partition_data[i].id)
                        else:
                            out[partition_data[j].id] = set([partition_data[i].id])
            return [Row(item[0], item[1]) for item in out.items()]
        return scan

    def __label(self):
        def label(x):
            if len(x[1]) + 1 >= self.min_pts:
                cluster_label = x[0]
                out = [(x[0], [(cluster_label, True)])]
                for idx in x[1]:
                    out.append((idx, [(cluster_label, False)]))
                return out
            return []
        return label

    def __combine_labels(self, x):
        point = x[0]
        core_point = False
        cluster_labels = x[1]
        clusters = []
        for (label, point_type) in cluster_labels:
            if point_type is True:
                core_point = True
            clusters.append(label)
        return point, clusters if core_point is True else [clusters[0]], core_point
