import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from selector.SupervisedFeatureSelectorABC import SupervisedFeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker


class RFESelector(SupervisedFeatureSelectorABC):
    """
    RFESelectorクラスは、再帰的特徴量削減に基づいて特徴量を選択する
    モデルを用いて再帰的に特徴量を削除し、最も重要な特徴量を選択

    Attributes:
        n_features_to_select (int): 選択する特徴量の数
    """

    @ArgsChecker((None, int), None)
    def __init__(self, n_features_to_select: int):
        self.n_features_to_select = n_features_to_select
        self.feature_names = None
        self.model = LogisticRegression(max_iter=10000)  # max_iter を増やす

    @ArgsChecker((None, pd.DataFrame, str), pd.DataFrame)
    def select_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # データのスケーリング
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rfe = RFE(self.model, n_features_to_select=self.n_features_to_select)
        rfe.fit(X_scaled, y)
        top_features = X.columns[rfe.support_]
        self.feature_names = top_features.tolist()
        return df[top_features]
