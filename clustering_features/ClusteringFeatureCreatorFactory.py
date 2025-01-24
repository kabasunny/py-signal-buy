# clustering _features\ClusteringFeatureCreatorFactory.py
from .PriceClusteringFeatureCreator import PriceClusteringFeatureCreator
from .VolumeClusteringFeatureCreator import VolumeClusteringFeatureCreator
from .PeakTroughClusteringFeatureCreator import PeakTroughClusteringFeatureCreator
from .FourierClusteringFeatureCreator import FourierClusteringFeatureCreator


class ClusteringFeatureCreatorFactory:
    """
    指定された特徴量作成クラスのインスタンスを生成するファクトリクラス。
    """

    @staticmethod
    def create_feature_creators(feature_list_str):
        creator_mapping = {
            "price_movement": PriceClusteringFeatureCreator,
            "volume": VolumeClusteringFeatureCreator,
            "peak_trough": PeakTroughClusteringFeatureCreator,
            "fourier": FourierClusteringFeatureCreator,
        }
        return [
            creator_mapping[feature]()
            for feature in feature_list_str
            if feature in creator_mapping
        ]
