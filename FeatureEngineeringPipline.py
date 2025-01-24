from features.FeatureCreationStage import FeatureCreationStage
from extractor.FeatureExtractionStage import FeatureExtractionStage
from selector.FeatureSelectionStage import FeatureSelectionStage
from features.FeatureCreatorFactory import FeatureCreatorFactory
from extractor.ExtractorFactory import ExtractorFactory
from selector.SelectorFactory import SelectorFactory
import time


class FeatureEngineeringPipline:
    def __init__(
        self,
        feature_period_days,  # 特徴量生成に必要な日数
        feature_list_str,
        data_managers,
        extractors,
        selectors,  # 新しい引数を追加
    ):
        self.feature_period_days = feature_period_days
        self.feature_list_str = feature_list_str

        self.data_managers = data_managers
        self.extractors = extractors
        self.selectors = selectors

        # 各パイプラインをインスタンス変数として保持
        self.feature_create_stage = FeatureCreationStage(
            self.data_managers["processed_raw"],
            self.data_managers["normalized_feature"],
            self.feature_period_days,
            FeatureCreatorFactory.create_feature_creators(self.feature_list_str),
        )
        self.extractor_stage = FeatureExtractionStage(
            self.data_managers["labeled"],
            self.data_managers["normalized_feature"],
            self.data_managers["extracted_feature"],
            ExtractorFactory.create_extractors(self.extractors),
        )
        self.selector_stage = FeatureSelectionStage(
            self.data_managers["labeled"],
            self.data_managers["normalized_feature"],
            self.data_managers["extracted_feature"],
            self.data_managers["selected_feature"],
            SelectorFactory.create_selectors(self.selectors),
        )

    def process_symbol(self, symbol):
        print(f"<< Now processing symbol {symbol} in {self.__class__.__name__} >>")

        try:
            stages = [
                ("FeatureCreationStage", self.feature_create_stage),
                ("FeatureExtractionStage", self.extractor_stage),
                ("FeatureSelectionStage", self.selector_stage),
            ]

            for stage_name, stage in stages:
                start_time = time.time()
                stage.run(symbol)
                elapsed_time = time.time() - start_time
                print(f"処理時間: {elapsed_time:.4f} 秒, {stage_name} ")

        except Exception as e:
            print(f"{symbol} の {stage_name} 処理中にエラーが発生しました: {e}")
