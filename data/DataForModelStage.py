from data.DataForModelPreparation import DataForModelPreparation
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.DataManager import DataManager
import pandas as pd


class DataForModelStage:
    @ArgsChecker(
        (
            None,
            DataManager,
            DataManager,
            DataManager,
            DataManager,
            str,  # 日付を受け取る
        ),
        None,
    )
    def __init__(
        self,
        label_data_manager: DataManager,
        selected_feature_manager: DataManager,
        training_and_test_data_manager: DataManager,
        practical_data_manager: DataManager,
        split_date: str,
    ):
        self.label_data_manager = label_data_manager
        self.selected_feature_manager = selected_feature_manager
        self.training_and_test_data_manager = training_and_test_data_manager
        self.practical_data_manager = practical_data_manager
        self.split_date = pd.to_datetime(split_date)  # 日付を格納

    @ArgsChecker((None, str), None)
    def run(self, symbol: str) -> None:
        # 特徴量データを読み込む
        full_data = self.selected_feature_manager.load_data(symbol)
        # データフレームが空でないことを確認
        if full_data.empty:
            print(f" {symbol} をスキップします")
            return

        # date カラムを Timestamp 型に変換
        if "date" in full_data.columns:
            full_data["date"] = pd.to_datetime(full_data["date"], errors="coerce")

        # ラベルデータを読み込んでマージ
        label_data = self.label_data_manager.load_data(symbol)
        # データフレームが空でないことを確認
        if label_data.empty:
            print(f" {symbol} のラベルデータが空です")
            return
        if "date" in label_data.columns:
            label_data["date"] = pd.to_datetime(label_data["date"], errors="coerce")

        # ラベルデータのマージ
        full_data = full_data.merge(
            label_data[["date", "label"]],
            on="date",
            how="left",
        )

        # split_date 以前のデータを抽出
        training_and_test_data = full_data[full_data["date"] <= self.split_date]

        t_t_correct_data, t_t_incorrect_data = (
            DataForModelPreparation.split_data_by_label(training_and_test_data)
        )

        # 訓練データとテストデータを準備
        combined_data = DataForModelPreparation.prepare_training_and_test_data(
            t_t_correct_data,
            t_t_incorrect_data,
            test_size=0.05,  # 必要に応じて調整
        )

        # 訓練データとテストデータを保存
        self.training_and_test_data_manager.save_data(combined_data, symbol)

        # split_date より後のデータを抽出
        practical_data = full_data[full_data["date"] > self.split_date]

        # 実践テストデータを保存
        self.practical_data_manager.save_data(practical_data, symbol)

        # print("Data For Model Pipeline completed successfully")
