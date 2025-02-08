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
        atr_period = 30  # 余裕を持って30個分のデータを使用

        for index, row in symbols.iterrows():
            symbol = row["symbol"]  # symbol 列の値を取得
            raw_data_df = self.raw_data_manager.load_data(symbol)

            if raw_data_df.empty:
                print(f" {symbol} をスキップします")
                continue

            # データの取得期間の確認
            # print(f"{symbol} のデータ期間: {raw_data_df['date'].min()} から {raw_data_df['date'].max()}")

            # ATR計算のために、split_date前の最新の30個分のデータを取得
            data_for_atr = raw_data_df[raw_data_df["date"] <= self.split_date].iloc[
                -atr_period:
            ]

            if data_for_atr.shape[0] < atr_period:
                print(
                    f"データが不足しています: {symbol} - {data_for_atr.shape[0]} 日分しかありません"
                )
                continue

            # デバッグ情報の追加
            # print(f"split_date: {self.split_date}")
            # print(f"{symbol} のデータ: {data_for_atr}")

            # ATR計算用のデータ
            daily_data_list_for_atr = [
                MLDailyData(
                    date=str(row["date"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                )
                for _, row in data_for_atr.iterrows()
            ]

            # split_date以降のデータも含める
            data_after_split_date = raw_data_df[raw_data_df["date"] > self.split_date]
            daily_data_list_after_split_date = [
                MLDailyData(
                    date=str(row["date"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                )
                for _, row in data_after_split_date.iterrows()
            ]

            # 全てのデータを連結
            daily_data_list = daily_data_list_for_atr + daily_data_list_after_split_date

            predictions_df = self.bach_pred_data_manager.load_data_from_subdir(
                subdir, symbol
            )

            if predictions_df.empty:
                print(f"データが見つからないため {symbol} をスキップします。")
                continue

            predictions_df = predictions_df[predictions_df["date"] > self.split_date]
            signal_dates = predictions_df[predictions_df["label"] == 1]["date"].tolist()

            # モデルのマップに正解ラベルのパターンを追加
            model_predictions = {
                "correct_label": ModelPredictions(prediction_dates=signal_dates)
            }

            # モデルの結果のマップ
            ensemble_dates = set()
            for model in self.model_types:
                prediction_dates = predictions_df[predictions_df[model] == 1][
                    "date"
                ].tolist()
                model_predictions[model] = ModelPredictions(
                    prediction_dates=[str(date) for date in prediction_dates]
                )
                ensemble_dates.update(prediction_dates)

            # 重複を避けるため、日付を一つだけにする
            ensemble_dates = sorted(list(ensemble_dates))

            model_predictions["ensemble_label"] = ModelPredictions(
                prediction_dates=[str(date) for date in ensemble_dates]
            )

            symbol_data = MLSymbolData(
                symbol=symbol,
                daily_data=daily_data_list,
                signals=[str(signal) for signal in signal_dates],
                model_predictions=model_predictions,
                priority=index,  # 優先順位をインデックスに基づいて設定
                split_date=self.split_date,  # split_date を追加
            )

            responses.append(symbol_data)  # MLSymbolData を直接追加

        combined_response = MLStockResponse(symbol_data=responses)

        self.proto_saver_loader.save_proto_response_to_file(
            combined_response,
            f'proto_{subdir.replace("/", "-")}.bin',
        )
