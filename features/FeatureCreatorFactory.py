# AnalyzerFactory.py
from features.PeakTroughFeatureCreator import PeakTroughFeatureCreator
from features.FourierFeatureCreator import FourierFeatureCreator
from features.VolumeFeatureCreator import VolumeFeatureCreator
from features.PriceFeatureCreator import PriceFeatureCreator
from features.PastDataFeatureCreator import PastDataFeatureCreator


class FeatureCreatorFactory:
    @staticmethod
    def create_feature_creators(feature_list_str):
        creator_mapping = {
            "peak_trough": PeakTroughFeatureCreator,
            "fourier": FourierFeatureCreator,
            "volume": VolumeFeatureCreator,
            "price": PriceFeatureCreator,
            "past": PastDataFeatureCreator,
        }
        return [
            creator_mapping[feature]()
            for feature in feature_list_str
            if feature in creator_mapping
        ]
