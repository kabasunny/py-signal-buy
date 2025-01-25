# AnalyzerFactory.py
from features.VolumeFeatureCreator import VolumeFeatureCreator
from features.PriceFeatureCreator import PriceFeatureCreator
from features.PastDataFeatureCreator import PastDataFeatureCreator


class FeatureCreatorFactory:
    @staticmethod
    def create_feature_creators(feature_list_str):
        creator_mapping = {
            "volume": VolumeFeatureCreator,
            "price": PriceFeatureCreator,
            "past": PastDataFeatureCreator,
        }
        return [
            creator_mapping[feature]()
            for feature in feature_list_str
            if feature in creator_mapping
        ]
