import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from extractor.SupervisedFeatureExtractorABC import SupervisedFeatureExtractorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PCRFeatureExtractor(SupervisedFeatureExtractorABC):
    @ArgsChecker((None, int), None)
    def __init__(self, n_components: int):
        self.pca = PCA(n_components=n_components)
        self.regression = LinearRegression()
        self.feature_names = None

    @ArgsChecker((None, pd.DataFrame, str), pd.DataFrame)
    def extract_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        features = df.drop(columns=[target_column])
        target = df[target_column]

        pca_features = self.pca.fit_transform(features)
        self.regression.fit(pca_features, target)

        self.feature_names = [f"PCR{i+1}" for i in range(pca_features.shape[1])]
        selected_df = pd.DataFrame(pca_features, columns=self.feature_names)
        return selected_df
