import pandas as pd
import numpy as np  # numpyをインポート
from features.FeatureCreatorABC import FeatureCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PastDataFeatureCreator(FeatureCreatorABC):
    @ArgsChecker((None, pd.DataFrame, pd.Timestamp), pd.DataFrame)
    def create_features(
        self, df: pd.DataFrame, trade_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        過去のデータに基づいて特徴量を生成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            trade_start_date (pd.Timestamp): トレード開始日

        Returns:
            pd.DataFrame: 特徴量が追加されたデータフレーム
        """
        # 1個前から5個前のデータに基づく特徴量を追加
        for i in range(1, 6):
            df[f"lag_{i}_close"] = df["close"].shift(i)
            df[f"lag_{i}_indicator"] = (df[f"lag_{i}_close"] > df["close"]).astype(int)
            df.drop(
                columns=[f"lag_{i}_close"], inplace=True
            )  # セレクターによる除外率が高い

        # コメントアウトされた特徴量
        # """
        # 価格の変動幅に関する特徴量
        df["daily_range"] = df["high"] - df["low"]
        df["close_open_diff"] = df["close"] - df["open"]
        df["prev_close_diff"] = df["close"] - df["close"].shift(1)
        df["range_ratio"] = df["daily_range"] / df["close"].shift(1)

        # 価格の方向性に関する特徴量
        df["price_up"] = (df["close"] > df["close"].shift(1)).astype(int)
        df["price_down"] = (df["close"] < df["close"].shift(1)).astype(int)
        df["consecutive_up"] = (
            (df["close"] > df["close"].shift(1))
            .astype(int)
            .groupby((df["close"] <= df["close"].shift(1)).astype(int).cumsum())
            .cumsum()
        )
        df["consecutive_up"] = df["consecutive_up"] * df["price_up"]

        df["consecutive_down"] = (
            (df["close"] < df["close"].shift(1))
            .astype(int)
            .groupby((df["close"] >= df["close"].shift(1)).astype(int).cumsum())
            .cumsum()
        )
        df["consecutive_down"] = df["consecutive_down"] * df["price_down"]

        # 特定の価格との比較に関する特徴量
        df["close_high_ratio"] = df["close"] / df["high"]
        df["close_low_ratio"] = df["close"] / df["low"]
        # """

        # 無限大の値をNaNに置き換える
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # NaNの値を補間する（例：直前の有効値で補間）
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # dfがまだNaNを含む場合、平均値で補完
        for column in df.columns:
            if df[column].isnull().any():
                df[column].fillna(df[column].mean(), inplace=True)

        return df
