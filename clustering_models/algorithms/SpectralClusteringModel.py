import pandas as pd
from sklearn.cluster import SpectralClustering
from clustering_models.ClusteringBaseModelABC import ClusteringBaseModelABC


class SpectralClusteringModel(ClusteringBaseModelABC):
    """
    スペクトルクラスタリングアルゴリズムを使用してデータをクラスタリングするクラス。
    """

    def __init__(self, n_clusters=5, affinity="nearest_neighbors", random_state=42):
        self.spectral = SpectralClustering(
            n_clusters=n_clusters, affinity=affinity, random_state=random_state
        )

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        入力データフレームをクラスタリングし、クラスタラベルを返すメソッド。

        Args:
            df (pd.DataFrame): クラスタリングするための入力データフレーム

        Returns:
            pd.Series: クラスタラベルを含むシリーズ
        """
        clusters = self.spectral.fit_predict(df)
        return pd.Series(clusters, index=df.index)
