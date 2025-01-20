import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from extractor.SupervisedFeatureExtractorABC import SupervisedFeatureExtractorABC
from decorators.ArgsChecker import ArgsChecker


class LDAFeatureExtractor(SupervisedFeatureExtractorABC):
    @ArgsChecker((None, int), None)
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.lda = None
        self.feature_names = None

    @ArgsChecker((None, pd.DataFrame, str), pd.DataFrame)
    def extract_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        features = df.drop(columns=[target_column])
        target = df[target_column]

        # LDAの n_components をクラス数と特徴量数に基づいて調整
        n_classes = target.nunique()
        n_features = features.shape[1]
        max_components = min(n_features, n_classes - 1)

        if self.n_components > max_components:
            print(
                f"Adjusting n_components from {self.n_components} to {max_components}"
            )
            self.n_components = max_components

        self.lda = LDA(n_components=self.n_components)
        self.lda.fit(features, target)
        extracted_features = self.lda.transform(features)
        self.feature_names = [f"LD{i+1}" for i in range(extracted_features.shape[1])]

        selected_df = pd.DataFrame(extracted_features, columns=self.feature_names)
        return selected_df
