from data.DataForModelStage import DataForModelStage
from models.ModelTrainStage import ModelTrainStage
import time


class ModelTrainingPipeline:
    def __init__(
        self,
        before_period_days,  # 特徴量生成に必要な日数
        split_date,  # トレーニング最終日、翌日以降実践日
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
    ):
        self.before_period_days = before_period_days
        self.split_date = split_date
        self.model_types = model_types
        self.feature_list_str = feature_list_str
        self.model_saver_loader = model_saver_loader
        self.model_created = False  # モデルが作成済みかどうかのフラグ
        self.data_managers = data_managers

        # 各ステージをインスタンス変数として保持
        self.data_for_model_stage = DataForModelStage(
            self.data_managers["labeled"],
            self.data_managers["selected_feature"],
            self.data_managers["training_and_test"],
            self.data_managers["practical"],
            self.split_date,
        )
        self.model_stage = ModelTrainStage(
            self.data_managers["training_and_test"],
            self.model_saver_loader,
            self.model_types,
        )

    def process_symbol(self, symbol, subdir):
        print(f"<< Now processing symbol {symbol} in {self.__class__.__name__} >>")
        try:
            start_time = time.time()
            self.data_for_model_stage.run(symbol)
            elapsed_time = time.time() - start_time
            print(f"処理時間: {elapsed_time:.4f} 秒, DataForModelStage ")

            start_time = time.time()
            self.model_stage.run(symbol, subdir)
            elapsed_time = time.time() - start_time
            print(f"処理時間: {elapsed_time:.4f} 秒, ModelTrainStage ")

        except Exception as e:
            print(f"{symbol} の処理中にエラーが発生しました: {e}")
