import pandas as pd
from sklearn.cluster import DBSCAN
from clustering_models.ClusteringBaseModelABC import ClusteringBaseModelABC


class DBSCANClustering(ClusteringBaseModelABC):
    """
    DBSCANクラスタリングアルゴリズムを使用してデータをクラスタリングするクラス。
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        入力データフレームをクラスタリングし、クラスタラベルを返すメソッド。

        Args:
            df (pd.DataFrame): クラスタリングするための入力データフレーム

        Returns:
            pd.Series: クラスタラベルを含むシリーズ
        """
        clusters = self.dbscan.fit_predict(df)
        return pd.Series(clusters, index=df.index)
