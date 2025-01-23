import pandas as pd
from sklearn.cluster import KMeans
from clustering_models.ClusteringBaseModelABC import ClusteringBaseModelABC


class KMeansClustering(ClusteringBaseModelABC):
    """
    K-meansクラスタリングアルゴリズムを使用してデータをクラスタリングするクラス。
    """

    def __init__(self, n_clusters=5, random_state=42):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        入力データフレームをクラスタリングし、クラスタラベルを返すメソッド。

        Args:
            df (pd.DataFrame): クラスタリングするための入力データフレーム

        Returns:
            pd.Series: クラスタラベルが追加されたデータ
        """
        clusters = self.kmeans.fit_predict(df)
        return clusters
