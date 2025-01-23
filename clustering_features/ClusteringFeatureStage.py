import pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocessing.Normalizer import Normalizer  # Normalizerクラスをインポート
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.DataManager import DataManager


class ClusteringFeatureStage:
    """
    特徴量生成の全体の流れを管理し、各特徴量作成クラスを呼び出すクラス。
    """

    @ArgsChecker((None, DataManager, DataManager, int, list), None)
    def __init__(
        self,
        processed_data_manager: DataManager,
        norm_ft_for_cluster: DataManager,
        before_period_days: int,
        feature_creators: list,
    ):
        self.normalizer = Normalizer()  # Normalizerクラスのインスタンスを作成
        self.processed_data_manager = processed_data_manager
        self.norm_ft_for_cluster = norm_ft_for_cluster
        self.feature_creators = feature_creators
        self.before_period_days = before_period_days

    @ArgsChecker((None, str), None)
    def run(self, symbol: str):
        """
        特徴量作成と選択を一連の流れで実行するメソッド

        Args:
            symbol (str): シンボル
        """
        # データをロード
        df = self.processed_data_manager.load_data(symbol)

        # データフレームの内容を詳細に表示
        # print(f"Loaded data for symbol {symbol}:\n", df.head())

        # データフレームが空でないことを確認
        if df.empty:
            print(f"{symbol} をスキップします")
            return

        # trade_start_day を計算
        first_date = pd.to_datetime(df["date"].iloc[0])
        trade_start_date = first_date + pd.DateOffset(days=self.before_period_days)

        # 特徴量を作成
        df["date"] = pd.to_datetime(df["date"])

        for creator in self.feature_creators:
            df = creator.create_features(df, trade_start_date)

        # 特徴量が生成された後のデータフレームを表示
        # print(f"Data after feature creation:\n", df.head())

        # trade_start_date 以降の日付のデータをフィルタリング
        df_with_features = df[df["date"] >= trade_start_date].copy()
        # print(f"Filtered data:\n", df_with_features.head())

        # 指定された列を削除
        columns_to_drop = ["open", "high", "low", "close", "volume"]
        df_with_features.drop(columns=columns_to_drop, inplace=True)

        # 正規化する列を自動で指定
        columns_to_normalize = df_with_features.columns.difference(
            ["date", "symbol"]
        ).tolist()

        # 特徴量を正規化
        df_normalized = self.normalizer.normalize(
            df_with_features, columns_to_normalize
        )
        # print(f"Normalized features:\n", df_normalized.head())

        # データを保存
        self.norm_ft_for_cluster.save_data(df_normalized, symbol)

        # print("Feature creation pipeline completed successfully")
