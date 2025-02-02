import pandas as pd
from sklearn.cluster import KMeans
from clustering_models.ClusteringBaseModelABC import ClusteringBaseModelABC
from sklearn.impute import SimpleImputer


class KMeansClustering(ClusteringBaseModelABC):
    """
    K-meansクラスタリングアルゴリズムを使用してデータをクラスタリングするクラス。
    """

    def __init__(self, n_clusters=1, random_state=42, init="k-means++", n_init=10):
        self.kmeans = KMeans(
            n_clusters=n_clusters, random_state=random_state, init=init, n_init=n_init
        )

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        入力データフレームをクラスタリングし、クラスタラベルを返すメソッド。

        Args:
            df (pd.DataFrame): クラスタリングするための入力データフレーム

        Returns:
            pd.Series: クラスタラベルが追加されたデータ
        """
        # NaN値を補完するためのイムプターを定義（平均値補完）
        imputer = SimpleImputer(strategy='mean')
        df_imputed = imputer.fit_transform(df)

        # クラスタリングを実行
        clusters = self.kmeans.fit_predict(df_imputed)
        return clusters
