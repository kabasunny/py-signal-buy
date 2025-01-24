from clustering_models.algorithms.KMeansClustering import KMeansClustering
from clustering_models.algorithms.DBSCANClustering import DBSCANClustering
from clustering_models.algorithms.HierarchicalClustering import HierarchicalClustering
from clustering_models.algorithms.MeanShiftClustering import MeanShiftClustering
from clustering_models.algorithms.GaussianMixtureClustering import (
    GaussianMixtureClustering,
)
from clustering_models.algorithms.SpectralClusteringModel import SpectralClusteringModel


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
        elif model_type == "hierarchical":
            return HierarchicalClustering()
        elif model_type == "mean_shift":
            return MeanShiftClustering()
        elif model_type == "gmm":
            return GaussianMixtureClustering()
        elif model_type == "spectral":
            return SpectralClusteringModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
