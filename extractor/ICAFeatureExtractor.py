import pandas as pd
from sklearn.decomposition import FastICA
from extractor.UnsupervisedFeatureExtractorABC import UnsupervisedFeatureExtractorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class ICAFeatureExtractor(UnsupervisedFeatureExtractorABC):
    @ArgsChecker((None, int), None)
    def __init__(self, n_components: int):
        self.ica = FastICA(n_components=n_components)
        self.feature_names = None

    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.ica.fit(df)
        extracted_features = self.ica.transform(df)
        self.feature_names = [f"IC{i+1}" for i in range(extracted_features.shape[1])]

        selected_df = pd.DataFrame(extracted_features, columns=self.feature_names)
        return selected_df
