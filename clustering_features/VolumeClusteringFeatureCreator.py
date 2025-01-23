import logging
import pandas as pd
from .ClusteringFeatureCreatorABC import ClusteringFeatureCreatorABC


class VolumeClusteringFeatureCreator(ClusteringFeatureCreatorABC):
    """
    出来高に関する特徴量を生成するクラス。
    """

    def __init__(self, volume_ma_period=20):
        self.volume_ma_period = volume_ma_period

    def create_features(
        self, df: pd.DataFrame, trade_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        出来高に関する特徴量を生成するメソッド。
        """
        try:
            df = df.copy()
            # print(f"Data before creating volume features:\n", df.head())

            if "volume" not in df.columns:
                raise ValueError("'volume' 列が存在しません")

            df["volume_change_rate"] = self.calculate_volume_change_rate(df)
            df["volume_moving_average"] = self.calculate_volume_moving_average(df)

            # NaN や inf を含む行を削除
            df.replace([float("inf"), -float("inf")], float("nan"), inplace=True)
            df.dropna(inplace=True)

            # print(f"Data after creating volume features:\n", df.head())

            return df
        except Exception as e:
            logging.error(f"Error in creating volume features: {e}")
            return pd.DataFrame()

    def calculate_volume_change_rate(self, df: pd.DataFrame) -> pd.Series:
        """
        出来高の増減率を計算するメソッド。
        """
        df["volume_change_rate"] = df["volume"].pct_change()
        return df["volume_change_rate"]

    def calculate_volume_moving_average(self, df: pd.DataFrame) -> pd.Series:
        """
        出来高の移動平均を計算するメソッド。
        """
        df["volume_moving_average"] = (
            df["volume"].rolling(window=self.volume_ma_period).mean()
        )
        return df["volume_moving_average"]
