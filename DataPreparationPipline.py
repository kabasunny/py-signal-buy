from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher
from data.DataAcquisitionAndFormattingStage import DataAcquisitionAndFormattingStage
from preprocessing.DataPreprocessingStage import DataPreprocessingStage
from labeling.LabelCreationStage import LabelCreationStage
from labeling.TroughLabelCreator import TroughLabelCreator
import time


class DataPreparationPipline:
    def __init__(
        self,
        feature_period_days,  # 特徴量生成に必要な日数
        data_managers,
    ):
        self.feature_period_days = feature_period_days
        self.model_created = False  # モデルが作成済みかどうかのフラグ

        self.data_managers = data_managers

        # 各ステージをインスタンス変数として保持
        self.raw_data_stage = DataAcquisitionAndFormattingStage(
            self.data_managers["formated_raw"],
            fetcher=YahooFinanceStockDataFetcher(),
        )
        self.preprocess_stage = DataPreprocessingStage(
            self.data_managers["formated_raw"], self.data_managers["processed_raw"]
        )
        self.label_create_stage = LabelCreationStage(
            self.data_managers["formated_raw"],
            self.data_managers["labeled"],
            self.feature_period_days,
            TroughLabelCreator(),
        )

    def process_symbol(self, symbol):
        print(f"<< Now processing symbol {symbol} in {self.__class__.__name__} >>")

        try:
            stages = [
                ("DataAcquisitionAndFormattingStage", self.raw_data_stage),
                ("DataPreprocessingStage", self.preprocess_stage),
                ("LabelCreationStage", self.label_create_stage),
            ]

            for stage_name, stage in stages:
                start_time = time.time()
                stage.run(symbol)
                elapsed_time = time.time() - start_time
                print(f"処理時間: {elapsed_time:.4f} 秒, {stage_name} ")

        except Exception as e:
            print(f"{symbol} の {stage_name} 処理中にエラーが発生しました: {e}")
