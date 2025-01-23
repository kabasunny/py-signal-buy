# clustering _features\ClusteringFeatureCreatorABC.py
from abc import ABC, abstractmethod
import pandas as pd


class ClusteringFeatureCreatorABC(ABC):
    """
    特徴量作成クラスの共通インターフェースを定義する抽象基底クラス。
    """

    @abstractmethod
    def create_features(
        self, df: pd.DataFrame, before_period_days: int
    ) -> pd.DataFrame:
        """
        特徴量を作成するための抽象メソッド。

        Args:
            df (pd.DataFrame): 入力データフレーム
            before_period_days (int): 特徴量生成に必要な日数

        Returns:
            pd.DataFrame: 特徴量が追加されたデータフレーム
        """
        pass
