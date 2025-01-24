import pandas as pd
from sklearn.cluster import MeanShift
from clustering_models.ClusteringBaseModelABC import ClusteringBaseModelABC
from sklearn.impute import SimpleImputer


class MeanShiftClustering(ClusteringBaseModelABC):
    """
    平均シフトクラスタリングアルゴリズムを使用してデータをクラスタリングするクラス。
    """

    def __init__(self):
        self.mean_shift = MeanShift()

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
        clusters = self.mean_shift.fit_predict(df_imputed)
        return pd.Series(clusters, index=df.index)
