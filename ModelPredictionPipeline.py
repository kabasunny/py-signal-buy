from result.ResultSavingStage import ResultSavingStage
from prediction.PredictionStage import PredictionStage
import time
from result.print_ml_stock_response import print_ml_stock_response_summary


class ModelPredictionPipeline:
    def __init__(
        self,
        before_period_days,  # 特徴量生成に必要な日数
        split_date,
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
        selectors,
        proto_saver_loader,
    ):
        self.before_period_days = before_period_days
        self.split_date = split_date
        self.model_types = model_types
        self.feature_list_str = feature_list_str
        self.model_saver_loader = model_saver_loader
        self.model_created = False
        self.data_managers = data_managers
        self.selectors = selectors
        self.proto_saver_loader = proto_saver_loader

        # 各パイプラインをインスタンス変数として保持
        self.real_predict_stage = PredictionStage(
            self.model_saver_loader,
            self.data_managers["practical"],
            self.data_managers["predictions"],
            self.model_types,
        )

        # ProtoConvertPipelineの初期化と実行
        self.proto_convert_stage = ResultSavingStage(
            self.data_managers["formated_raw"],
            self.data_managers["predictions"],
            self.proto_saver_loader,
            self.model_types,
            self.split_date,
        )

    def process_symbol(self, symbol):
        print(f"Symbol of current data: {symbol}")

        try:
            stages = [
                ("PredictionStage", self.real_predict_stage),
            ]

            for stage_name, stage in stages:
                start_time = time.time()
                stage.run(symbol)
                # print(f"\n symbol : {symbol} \n")
                elapsed_time = time.time() - start_time
                print(f"{stage_name} processing time: {elapsed_time:.4f} 秒")

        except Exception as e:
            print(f"{symbol} の {stage_name} 処理中にエラーが発生しました: {e}")

    def finish_prosess(self, symbols):
        print(f"Proto file processing for all symbols")
        # print(symbols)
        self.proto_convert_stage.run(
            symbols
        )  # リストを受けるため他のパイプラインと異なる

        # # 保存したプロトコルバッファーの読み込み
        # loaded_proto_response = self.proto_saver_loader.load_proto_response_from_file()
        # print_ml_stock_response_summary(loaded_proto_response)
