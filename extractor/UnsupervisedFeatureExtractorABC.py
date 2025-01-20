from abc import ABC, abstractmethod
import pandas as pd

from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class UnsupervisedFeatureExtractorABC(ABC):
    @abstractmethod
    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量を選択するための抽象メソッド（正解ラベル不要）

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        pass
