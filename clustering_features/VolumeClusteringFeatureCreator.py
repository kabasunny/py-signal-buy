import logging
import pandas as pd
from .ClusteringFeatureCreatorABC import ClusteringFeatureCreatorABC


class VolumeClusteringFeatureCreator(ClusteringFeatureCreatorABC):
    """
    出来高の変化に関する特徴量を生成するクラス。
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

            if "volume" not in df.columns:
                raise ValueError("'volume' 列が存在しません")

            df["vol_chg"] = self.calculate_volume_change_rate(df)  # 出来高の増減率
            df["vol_ma"] = self.calculate_volume_moving_average(df)  # 出来高の移動平均

            # NaN や無限大を含む値を処理
            df.replace([float("inf"), -float("inf")], float("nan"), inplace=True)
            df.dropna(inplace=True)  # 必要に応じて補完することもできます

            # 統計指数を計算
            stats = self.calculate_statistics(df)

            return stats
        except Exception as e:
            logging.error(f"Error in creating volume features: {e}")
            return pd.DataFrame()

    def calculate_volume_change_rate(self, df: pd.DataFrame) -> pd.Series:
        """
        出来高の増減率を計算するメソッド。
        """
        df["vol_chg"] = df["volume"].pct_change()
        return df["vol_chg"]

    def calculate_volume_moving_average(self, df: pd.DataFrame) -> pd.Series:
        """
        出来高の移動平均を計算するメソッド。
        """
        df["vol_ma"] = df["volume"].rolling(window=self.volume_ma_period).mean()
        return df["vol_ma"]

    def calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        統計指数を計算するメソッド。
        """
        stats = {
            "vol_chg_mean": df["vol_chg"].mean(),  # 出来高増減率の平均
            "vol_chg_std": df["vol_chg"].std(),  # 出来高増減率の標準偏差
            "vol_ma_mean": df["vol_ma"].mean(),  # 出来高移動平均の平均
            "vol_ma_std": df["vol_ma"].std(),  # 出来高移動平均の標準偏差
        }
        return pd.DataFrame([stats])
