from abc import ABC, abstractmethod
import pandas as pd


class ClusteringBaseModelABC(ABC):
    """
    クラスタリングモデルのための抽象基底クラス。
    """

    @abstractmethod
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        クラスタリングを実行し、クラスタラベルを追加するメソッド。

        Args:
            df (pd.DataFrame): クラスタリングするための入力データフレーム

        Returns:
            pd.DataFrame: クラスタラベルが追加されたデータフレーム
        """
        pass
