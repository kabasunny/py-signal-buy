import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from clustering_models.ClusteringBaseModelABC import ClusteringBaseModelABC


class HierarchicalClustering(ClusteringBaseModelABC):
    """
    層別クラスタリングアルゴリズムを使用してデータをクラスタリングするクラス。
    """

    def __init__(self, n_clusters=5):
        self.hierarchical = AgglomerativeClustering(n_clusters=n_clusters)

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        入力データフレームをクラスタリングし、クラスタラベルを返すメソッド。

        Args:
            df (pd.DataFrame): クラスタリングするための入力データフレーム

        Returns:
            pd.Series: クラスタラベルを含むシリーズ
        """
        clusters = self.hierarchical.fit_predict(df)
        return pd.Series(clusters, index=df.index)
