from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher
from data.DataAcquisitionAndFormattingStage import DataAcquisitionAndFormattingStage
from preprocessing.DataPreprocessingStage import DataPreprocessingStage
from labeling.LabelCreationStage import LabelCreatePipeline
from labeling.TroughLabelCreator import TroughLabelCreator
from features.FeatureEngineeringStage import FeatureEngineeringStage
from selector.FeatureSelectionStage import FeatureSelectionStage
from data.DataForModelPipeline import DataForModelPipeline
from features.AnalyzerFactory import AnalyzerFactory
from selector.SelectorFactory import SelectorFactory
from models.ModelPipeline import ModelPipeline
from models.ModelPredictPipeline import ModelPredictPipeline
import time



class TrainAutomatedPipeline:
    def __init__(
        self,
        before_period_days,  # 特徴量生成に必要な日数
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
        selectors  # 新しい引数を追加
    ):
        self.before_period_days = before_period_days
        self.model_types = model_types
        self.feature_list_str = feature_list_str
        self.model_saver_loader = model_saver_loader
        self.model_created = False  # モデルが作成済みかどうかのフラグ

        self.data_managers = data_managers
        self.selectors = selectors

        # 各パイプラインをインスタンス変数として保持
        self.raw_data_pipeline = DataAcquisitionAndFormattingStage(
            self.data_managers["formated_raw"],
            fetcher=YahooFinanceStockDataFetcher(),
        )
        self.preprocess_pipeline = DataPreprocessingStage(
            self.data_managers["formated_raw"], self.data_managers["processed_raw"]
        )
        self.label_create_pipeline = LabelCreatePipeline(
            self.data_managers["formated_raw"],
            self.data_managers["labeled"],
            self.before_period_days,
            TroughLabelCreator(),
        )
        self.feature_pipeline = FeatureEngineeringStage(
            self.data_managers["processed_raw"],
            self.data_managers["normalized_feature"],
            self.before_period_days,
            AnalyzerFactory.create_analyzers(self.feature_list_str),
        )
        self.selector_pipeline = FeatureSelectionStage(
            self.data_managers["labeled"],
            self.data_managers["normalized_feature"],
            self.data_managers["selected_feature"],
            self.data_managers["selected_ft_with_label"],
            SelectorFactory.create_selectors(self.selectors), 
        )
        self.data_for_model_pipeline = DataForModelPipeline(
            self.data_managers["labeled"],
            self.data_managers["selected_feature"],
            self.data_managers["training_and_test"],
            self.data_managers["practical"],
        )
        self.model_pipeline = ModelPipeline(
            self.data_managers["training_and_test"],
            self.model_saver_loader,
            self.model_types,
        )
        self.model_predict_pipeline = ModelPredictPipeline(
            self.model_saver_loader,
            self.data_managers["training_and_test"],
            self.data_managers["practical"],
            self.data_managers["predictions"],
            self.model_types,
        )

    def process_symbol(self, symbol):
        print(f"Symbol of current data: {symbol}")

        try:
            pipelines = [
                ("RawDataPipeline", self.raw_data_pipeline),
                ("PreprocessPipeline", self.preprocess_pipeline),
                ("LabelCreatePipeline", self.label_create_pipeline),
                ("FeaturePipeline", self.feature_pipeline),
                ("SelectorPipeline", self.selector_pipeline),
                ("DataForModelPipeline", self.data_for_model_pipeline),
                ("ModelPipeline", self.model_pipeline),
                ("ModelPredictPipeline", self.model_predict_pipeline),
            ]

            for pipeline_name, pipeline in pipelines:
                start_time = time.time()
                pipeline.run(symbol)
                elapsed_time = time.time() - start_time
                print(f"{pipeline_name} 処理時間: {elapsed_time:.4f} 秒")

        except Exception as e:
            print(f"{symbol} の処理中にエラーが発生しました: {e}")
