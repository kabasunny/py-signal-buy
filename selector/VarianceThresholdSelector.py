import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from selector.UnsupervisedFeatureSelectorABC import UnsupervisedFeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker


class VarianceThresholdSelector(UnsupervisedFeatureSelectorABC):
    """
    VarianceThresholdSelectorクラスは、分散閾値に基づいて特徴量を選択するセレクターです。
    一定の分散閾値を超える特徴量のみを選択します。

    Attributes:
        threshold (float): 分散の閾値
    """

    @ArgsChecker((None, float), None)
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.feature_names = None

    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        selector = VarianceThreshold(threshold=self.threshold)
        selector.fit(df)
        top_features = df.columns[selector.get_support()]
        self.feature_names = top_features.tolist()
        return df[top_features]
