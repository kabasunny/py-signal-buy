
from data.DataForModelStage import DataForModelStage
from models.ModelTrainStage import ModelTrainStage
from models.ModelPredictPipeline import ModelPredictPipeline
import time

class ModelTrainingPipeline:
    def __init__(
        self,
        before_period_days,  # 特徴量生成に必要な日数
        split_date, # トレーニング最終日、翌日以降実践日
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
        selectors  # 新しい引数を追加
    ):
        self.before_period_days = before_period_days
        self.split_date = split_date
        self.model_types = model_types
        self.feature_list_str = feature_list_str
        self.model_saver_loader = model_saver_loader
        self.model_created = False  # モデルが作成済みかどうかのフラグ
        self.data_managers = data_managers
        self.selectors = selectors

        # 各パイプラインをインスタンス変数として保持
        
        self.data_for_model_pipeline = DataForModelStage(
            self.data_managers["selected_ft_with_label"],
            self.data_managers["training_and_test"],
            self.data_managers["practical"],
            self.split_date,
            )
        self.model_pipeline = ModelTrainStage(
            self.data_managers["training_and_test"],
            self.model_saver_loader,
            self.model_types,
        )

    def process_symbol(self, symbol):
        print(f"Symbol of current data: {symbol}")

        try:
            pipelines = [
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
