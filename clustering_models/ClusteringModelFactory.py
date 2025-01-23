from clustering_models.algorithms.KMeansClustering import KMeansClustering
from clustering_models.algorithms.DBSCANClustering import DBSCANClustering


class ClusteringModelFactory:
    """
    指定されたクラスタリングモデルのインスタンスを生成するファクトリクラス。
    """

    @staticmethod
    def create_model(model_type: str):
        if model_type == "kmeans":
            return KMeansClustering()
        elif model_type == "dbscan":
            return DBSCANClustering()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
