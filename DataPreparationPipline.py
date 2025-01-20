from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher
from data.DataAcquisitionAndFormattingStage import DataAcquisitionAndFormattingStage
from preprocessing.DataPreprocessingStage import DataPreprocessingStage
from labeling.LabelCreationStage import LabelCreateStage
from labeling.TroughLabelCreator import TroughLabelCreator
from features.FeatureEngineeringStage import FeatureEngineeringStage
from extractor.FeatureExtractionStage import FeatureExtractionStage
from selector.FeatureSelectionStage import FeatureSelectionStage
from features.AnalyzerFactory import AnalyzerFactory
from extractor.ExtractorFactory import ExtractorFactory
from selector.SelectorFactory import SelectorFactory
import time


class DataPreparationPipline:
    def __init__(
        self,
        before_period_days,  # 特徴量生成に必要な日数
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
        extractors,
        selectors,  # 新しい引数を追加
    ):
        self.before_period_days = before_period_days
        self.model_types = model_types
        self.feature_list_str = feature_list_str
        self.model_saver_loader = model_saver_loader
        self.model_created = False  # モデルが作成済みかどうかのフラグ

        self.data_managers = data_managers
        self.extractors = extractors
        self.selectors = selectors

        # 各パイプラインをインスタンス変数として保持
        self.raw_data_stage = DataAcquisitionAndFormattingStage(
            self.data_managers["formated_raw"],
            fetcher=YahooFinanceStockDataFetcher(),
        )
        self.preprocess_stage = DataPreprocessingStage(
            self.data_managers["formated_raw"], self.data_managers["processed_raw"]
        )
        self.label_create_stage = LabelCreateStage(
            self.data_managers["formated_raw"],
            self.data_managers["labeled"],
            self.before_period_days,
            TroughLabelCreator(),
        )
        self.feature_stage = FeatureEngineeringStage(
            self.data_managers["processed_raw"],
            self.data_managers["normalized_feature"],
            self.before_period_days,
            AnalyzerFactory.create_analyzers(self.feature_list_str),
        )
        self.extractor_stage = FeatureExtractionStage(
            self.data_managers["labeled"],
            self.data_managers["normalized_feature"],
            self.data_managers["extracted_ft_with_label"],
            ExtractorFactory.create_extractors(self.extractors),
        )
        self.selector_stage = FeatureSelectionStage(
            self.data_managers["labeled"],
            self.data_managers["normalized_feature"],
            self.data_managers["extracted_ft_with_label"],
            self.data_managers["selected_ft_with_label"],
            SelectorFactory.create_selectors(self.selectors),
        )

    def process_symbol(self, symbol):
        print(f"Symbol of current data: {symbol}")

        try:
            stages = [
                ("DataAcquisitionAndFormattingStage", self.raw_data_stage),
                ("DataPreprocessingStage", self.preprocess_stage),
                ("LabelCreateStage", self.label_create_stage),
                ("FeatureEngineeringStage", self.feature_stage),
                ("FeatureExtractionStage", self.extractor_stage),
                ("FeatureSelectionStage", self.selector_stage),
            ]

            for stage_name, stage in stages:
                start_time = time.time()
                stage.run(symbol)
                elapsed_time = time.time() - start_time
                print(f"{stage_name} 処理時間: {elapsed_time:.4f} 秒")

        except Exception as e:
            print(f"{symbol} の {stage_name} 処理中にエラーが発生しました: {e}")
