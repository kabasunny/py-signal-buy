import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from selector.SupervisedFeatureSelectorABC import SupervisedFeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker


class MutualInformationSelector(SupervisedFeatureSelectorABC):
    """
    MutualInformationSelectorクラスは、相互情報量に基づいて特徴量を選択
    特徴量とターゲット変数との間の相互情報量を計算し、最も関連性の高い特徴量を選択

    Attributes:
        k (int): 選択するトップkの特徴量の数
    """

    @ArgsChecker((None, int), None)
    def __init__(self, k: int):
        self.k = k
        self.feature_names = None

    @ArgsChecker((None, pd.DataFrame, str), pd.DataFrame)
    def select_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        mi = mutual_info_classif(X, y)
        mi_series = pd.Series(mi, index=X.columns)
        top_features = mi_series.nlargest(self.k).index
        self.feature_names = top_features.tolist()
        return df[top_features]
