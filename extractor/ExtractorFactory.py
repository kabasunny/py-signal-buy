from extractor.LDAFeatureExtractor import LDAFeatureExtractor
from extractor.ICAFeatureExtractor import ICAFeatureExtractor
from extractor.PCRFeatureExtractor import PCRFeatureExtractor
from extractor.PCAFeatureExtractor import PCAFeatureExtractor


class ExtractorFactory:
    """
    ExtractorFactoryクラスは、各種特徴量抽出器を生成するためのファクトリクラス。
    """

    @staticmethod
    def create_extractors(extractor_names):
        extractors = []
        for extractor_name in extractor_names:
            if extractor_name == "PCA":
                extractors.append(PCAFeatureExtractor(n_components=2))
            elif extractor_name == "LDA":
                extractors.append(LDAFeatureExtractor(n_components=2))
            elif extractor_name == "ICA":
                extractors.append(ICAFeatureExtractor(n_components=2))
            elif extractor_name == "PCR":
                extractors.append(PCRFeatureExtractor(n_components=2))
            else:
                raise ValueError(f"Unknown extractor: {extractor_name}")
        
        # print(extractors)
        return extractors
