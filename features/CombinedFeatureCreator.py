import pandas as pd
from features.PeakTroughFeatureCreator import PeakTroughFeatureCreator
from features.FourierFeatureCreator import FourierFeatureCreator
from features.VolumeFeatureCreator import VolumeFeatureCreator
from features.PriceFeatureCreator import PriceFeatureCreator
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class CombinedFeatureCreator:
    @ArgsChecker((None, list, pd.Timestamp), None)
    def __init__(self, feature_list_str: list, trade_start_date: pd.Timestamp):
        creator_mapping = {
            "peak_trough": PeakTroughFeatureCreator,
            "fourier": FourierFeatureCreator,
            "volume": VolumeFeatureCreator,
            "price": PriceFeatureCreator,
        }
        self.creators = [
            creator_mapping[feature]()
            for feature in feature_list_str
            if feature in creator_mapping
        ]
        self.trade_start_date = trade_start_date

    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全ての特徴量を作成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 全ての特徴量が追加されたデータフレーム
        """
        # dateカラムをTimestamp型に変換
        df["date"] = pd.to_datetime(df["date"])

        for creator in self.creators:
            df = creator.create_features(df, self.trade_start_date)
            # print(f"Data with features: {analyzer}")
            # print(df.head())
            # print(df.tail())
            # print(df.info())

        # trade_start_date 以降の日付のデータをフィルタリング
        df = df[df["date"] >= self.trade_start_date].copy()

        return df
