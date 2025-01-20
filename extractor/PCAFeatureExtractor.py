import pandas as pd
from sklearn.decomposition import PCA
from extractor.UnsupervisedFeatureExtractorABC import UnsupervisedFeatureExtractorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PCAFeatureExtractor(UnsupervisedFeatureExtractorABC):
    """
    PCAFeatureSelectorクラスは、主成分分析（PCA）に基づいて特徴量を選択するためのクラス。
    元の高次元データを低次元の主成分に変換し、データの分散を最大限に保持しながら、重要な特徴量を抽出します。

    Attributes:
        pca (PCA): PCAモデル
        feature_names (list): 選択された特徴量の名前
    """

    @ArgsChecker((None, int), None)
    def __init__(self, n_components: int):
        self.pca = PCA(n_components=n_components)
        self.feature_names = None

    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PCAに基づいて特徴量を選択するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        self.pca.fit(df)
        extracted_features = self.pca.transform(df)
        self.feature_names = [f"PC{i+1}" for i in range(extracted_features.shape[1])]

        selected_df = pd.DataFrame(extracted_features, columns=self.feature_names)
        return selected_df
