from result.ml_stock_service_pb2 import (
    MLStockResponse,
    MLSymbolData,
    MLDailyData,
    ModelPredictions,
)
import pandas as pd
from typing import List
from data.DataManager import DataManager
from result.ProtoSaverLoader import ProtoSaverLoader
from result.print_ml_stock_response import print_ml_stock_response_summary


class ResultSavingStage:
    def __init__(
        self,
        raw_data_manager: DataManager,
        bach_pred_data_manager: DataManager,
        proto_saver_loader: ProtoSaverLoader,
        model_types: List[str],
        split_date: str,
    ):
        self.raw_data_manager = raw_data_manager
        self.bach_pred_data_manager = bach_pred_data_manager
        self.proto_saver_loader = proto_saver_loader
        self.model_types = model_types
        self.split_date = split_date  # 文字列のまま保存

    def run(
        self,
        symbols: pd.DataFrame,
        subdir: str,
    ):
        responses = []
        for index, row in symbols.iterrows():
            symbol = row["symbol"]  # symbol 列の値を取得
            raw_data_df = self.raw_data_manager.load_data(symbol)

            if raw_data_df.empty:
                print(f" {symbol} をスキップします")
                continue

            raw_data_df = raw_data_df[raw_data_df["date"] > self.split_date]

            daily_data_list = [
                MLDailyData(
                    date=str(row["date"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                )
                for _, row in raw_data_df.iterrows()
            ]

            predictions_df = self.bach_pred_data_manager.load_data_from_subdir(
                subdir, symbol
            )

            if predictions_df.empty:
                print(f"データが見つからないため {symbol} をスキップします。")
                continue

            predictions_df = predictions_df[predictions_df["date"] > self.split_date]

            signal_dates = predictions_df[predictions_df["label"] == 1]["date"].tolist()

            # モデルのマップに正解ラベルのパターンを追加した
            model_predictions = {
                "correct_label": ModelPredictions(prediction_dates=signal_dates)
            }

            # モデルの結果のマップ
            for model in self.model_types:
                prediction_dates = predictions_df[predictions_df[model] == 1][
                    "date"
                ].tolist()
                model_predictions[model] = ModelPredictions(
                    prediction_dates=[str(date) for date in prediction_dates]
                )

            symbol_data = MLSymbolData(
                symbol=symbol,
                daily_data=daily_data_list,
                signals=[str(signal) for signal in signal_dates],
                model_predictions=model_predictions,
                priority=index,  # 優先順位をインデックスに基づいて設定
            )

            responses.append(symbol_data)  # MLSymbolData を直接追加

        combined_response = MLStockResponse(symbol_data=responses)

        self.proto_saver_loader.save_proto_response_to_file(
            combined_response,
            f'proto_{subdir.replace("/", "-")}.bin',
        )

        
        # 保存したプロトコルバッファーの読み込み
        loaded_proto_response = self.proto_saver_loader.load_proto_response_from_file(f'proto_{subdir.replace("/", "-")}.bin',
)
        print_ml_stock_response_summary(loaded_proto_response)
