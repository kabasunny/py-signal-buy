import pandas as pd
from preprocessing.Normalizer import Normalizer  # Normalizerクラスをインポート
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.DataManager import DataManager


# 教師あり学習向け
class FeatureCreationStage:
    @ArgsChecker((None, DataManager, DataManager, int, list), None)
    def __init__(
        self,
        processed_data_manager: DataManager,
        normalized_f_d_manager: DataManager,
        feature_period_days: int,
        analyzers: list,
    ):
        self.normalizer = Normalizer()  # Normalizerクラスのインスタンスを作成
        self.processed_data_manager = processed_data_manager
        self.normalized_f_d_manager = normalized_f_d_manager
        self.analyzers = analyzers
        self.feature_period_days = feature_period_days

    @ArgsChecker((None, str), None)
    def run(self, symbol):
        """
        特徴量作成と選択を一連の流れで実行するメソッド

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        # print("Run Feature creation pipeline")
        # データをロード
        df = self.processed_data_manager.load_data(symbol)
        # データフレームが空でないことを確認
        if df.empty:
            print(f" {symbol} をスキップします")
            return
        # print(f"df3\n{df.head(1)}")
        # trade_start_day を計算
        first_date = pd.to_datetime(df["date"].iloc[0])
        ft_pred_start_date = first_date + pd.DateOffset(days=self.feature_period_days)

        # 特徴量を作成
        # dateカラムをTimestamp型に変換
        df["date"] = pd.to_datetime(df["date"])

        for analyzer in self.analyzers:
            df = analyzer.create_features(df, ft_pred_start_date)

        # trade_start_date 以降の日付のデータをフィルタリング
        df_with_features = df[df["date"] >= ft_pred_start_date].copy()
        # print(f"df_with_features\n{df_with_features.head(1)}")

        # 指定された列を削除
        columns_to_drop = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        df_with_features.drop(columns=columns_to_drop, inplace=True)

        # 正規化する列を自動で指定
        columns_to_normalize = df_with_features.columns.difference(
            ["date", "symbol"]
        ).tolist()

        # 特徴量を正規化
        df_normalized = self.normalizer.normalize(
            df_with_features, columns_to_normalize
        )

        self.normalized_f_d_manager.save_data(df_normalized, symbol)

        # print("Feature creation pipeline completed successfully")
