import pandas as pd
from sklearn.mixture import GaussianMixture
from clustering_models.ClusteringBaseModelABC import ClusteringBaseModelABC
from sklearn.impute import SimpleImputer


class GaussianMixtureClustering(ClusteringBaseModelABC):
    """
    ガウシアン混合モデル（GMM）を使用してデータをクラスタリングするクラス。
    """

    def __init__(self, n_components=5, random_state=42):
        self.gmm = GaussianMixture(n_components=n_components, random_state=random_state)

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
        clusters = self.gmm.fit_predict(df_imputed)
        return pd.Series(clusters, index=df.index)
