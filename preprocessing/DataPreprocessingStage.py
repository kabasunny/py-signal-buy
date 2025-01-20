from decorators.ArgsChecker import ArgsChecker
from preprocessing.MissingValueHandler import MissingValueHandler
from preprocessing.OutlierDetector import OutlierDetector
from preprocessing.Normalizer import Normalizer
from data.DataManager import DataManager


class DataPreprocessingStage:
    @ArgsChecker((None, DataManager, DataManager), None)
    def __init__(
        self,
        raw_data_manager: DataManager,
        processed_data_manager: DataManager,
    ):
        self.raw_data_manager = raw_data_manager
        self.processed_data_manager = processed_data_manager

    @ArgsChecker((None, str), None)
    def run(self, symbol):
        """データパイプラインの実行"""
        # print("Run Preprocess pipeline")
        # データの読み込み
        df = self.raw_data_manager.load_data(symbol)
        # print("Raw data loaded successfully")
        # データフレームが空でないことを確認
        if df.empty:
           print(f" {symbol} をスキップします")
           return

        # データの前処理
        df = MissingValueHandler.fill_missing_with_mean(df)
        # print("Handled missing values")

        outliers = OutlierDetector.detect_outliers(df)
        # print("Outliers detected")

        # 正規化する列を指定
        columns_to_normalize = ["open", "high", "low", "close", "volume"]

        # 正規化の適用
        df = Normalizer.normalize(df, columns_to_normalize)
        # print("Normalized data")

        # データの保存
        self.processed_data_manager.save_data(df, symbol)
        # print("Processed data saved successfully")

        # print("Preprocess pipeline completed successfully")
