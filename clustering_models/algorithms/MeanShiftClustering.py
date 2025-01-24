import pandas as pd
from sklearn.cluster import MeanShift
from clustering_models.ClusteringBaseModelABC import ClusteringBaseModelABC


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
        clusters = self.mean_shift.fit_predict(df)
        return pd.Series(clusters, index=df.index)
