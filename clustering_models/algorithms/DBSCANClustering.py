import pandas as pd
from sklearn.cluster import DBSCAN
from clustering_models.ClusteringBaseModelABC import ClusteringBaseModelABC
from sklearn.impute import SimpleImputer


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
        # NaN値を補完するためのイムプターを定義（平均値補完）
        imputer = SimpleImputer(strategy='mean')
        df_imputed = imputer.fit_transform(df)

        # クラスタリングを実行
        clusters = self.dbscan.fit_predict(df_imputed)
        return pd.Series(clusters, index=df.index)
