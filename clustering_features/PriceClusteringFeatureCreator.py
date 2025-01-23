# clustering _features\PriceClusteringFeatureCreator.py
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
            df["daily_return"] = self.calculate_daily_return(df)
            df["volatility"] = self.calculate_volatility(df)

            # print(df.head())
            # print(df[["daily_return", "volatility"]])

            return df
        except Exception as e:
            logging.error(f"Error in creating price movement features: {e}")
            return pd.DataFrame()

    def calculate_daily_return(self, df: pd.DataFrame) -> pd.Series:
        """
        日次リターンを計算するメソッド。
        """
        df["daily_return"] = df["close"].pct_change()
        return df["daily_return"]

    def calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        ボラティリティを計算するメソッド。
        """
        df["volatility"] = (
            df["daily_return"].rolling(window=self.volatility_period).std()
        )
        return df["volatility"]
