import logging
import pandas as pd
from .ClusteringFeatureCreatorABC import ClusteringFeatureCreatorABC


class PriceClusteringFeatureCreator(ClusteringFeatureCreatorABC):
    """
    株価の値動きに関する特徴量を生成するクラス。
    """

    def __init__(self, volatility_period=20):
        self.volatility_period = volatility_period

    def create_features(
        self, df: pd.DataFrame, before_period_days: int
    ) -> pd.DataFrame:
        """
        株価の値動きに関する特徴量を生成するメソッド。
        """
        try:
            df = df.copy()
            df["daily_ret"] = self.calculate_daily_return(df)  # 日次リターン
            df["vol"] = self.calculate_volatility(df)  # ボラティリティ
            df["ATR"] = self.calculate_atr(df)  # ATR

            # NaN や無限大を含む値を処理
            df.replace([float("inf"), -float("inf")], float("nan"), inplace=True)
            df.dropna(inplace=True)  # 必要に応じて補完することもできます

            # 統計指数を計算
            stats = self.calculate_statistics(df)

            return stats
        except Exception as e:
            logging.error(f"Error in creating price movement features: {e}")
            return pd.DataFrame()

    def calculate_daily_return(self, df: pd.DataFrame) -> pd.Series:
        """
        日次リターンを計算するメソッド。
        """
        df["daily_ret"] = df["close"].pct_change()
        return df["daily_ret"]

    def calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        ボラティリティを計算するメソッド。
        """
        df["vol"] = df["daily_ret"].rolling(window=self.volatility_period).std()
        return df["vol"]

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        ATR（平均真の範囲）を計算するメソッド。
        """
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range  # 最後に平均するので、1日の範囲で計算

    def calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        統計指数を計算するメソッド。
        """
        stats = {
            "daily_ret_mean": df["daily_ret"].mean(),  # 日次リターンの平均
            "daily_ret_std": df["daily_ret"].std(),  # 日次リターンの標準偏差
            "vol_mean": df["vol"].mean(),  # ボラティリティの平均
            "vol_std": df["vol"].std(),  # ボラティリティの標準偏差
            "ATR": df["ATR"].mean(),  # ATRの平均
        }
        return pd.DataFrame([stats])
