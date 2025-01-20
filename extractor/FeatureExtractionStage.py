import pandas as pd
from extractor.SupervisedFeatureExtractorABC import SupervisedFeatureExtractorABC
from extractor.UnsupervisedFeatureExtractorABC import UnsupervisedFeatureExtractorABC
from decorators.ArgsChecker import ArgsChecker
from data.DataManager import DataManager


class FeatureExtractionStage:
    @ArgsChecker((None, DataManager, DataManager, DataManager, list), None)
    def __init__(
        self,
        label_data_manager: DataManager,
        normalized_f_d_manager: DataManager,
        extracted_f_w_l_d_manager: DataManager,
        extractors: list,
    ):
        self.label_data_manager = label_data_manager
        self.normalized_f_d_manager = normalized_f_d_manager
        self.extracted_f_w_l_d_manager = extracted_f_w_l_d_manager
        self.extractors = extractors

    @ArgsChecker((None, str), None)
    def run(self, symbol) -> None:
        """
        特徴量抽出を一連の流れで実行するメソッド

        Args:
            symbol (str): 対象のシンボル名
        """
        # 正規化済みデータをロード
        df_normalized = self.normalized_f_d_manager.load_data(symbol)

        # date カラムを Timestamp 型に変換
        if "date" in df_normalized.columns:
            df_normalized["date"] = pd.to_datetime(df_normalized["date"])

        # ラベルデータを読み込んでマージ
        target_df = self.label_data_manager.load_data(symbol)
        # データフレームが空でないことを確認
        if target_df.empty:
           print(f" {symbol} をスキップします")
           return
        if "date" in target_df.columns:
            target_df["date"] = pd.to_datetime(target_df["date"])

        df_with_label = df_normalized.merge(
            target_df[["date", "label"]],
            on=["date"],
            how="left",
        )

        # symbol を一行目から取得して保存
        symbol_value = (
            df_normalized["symbol"].iloc[0]
            if "symbol" in df_normalized.columns
            else None
        )

        # 不要なカラムをドロップ
        columns_to_drop = [
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        df_with_label = df_with_label.drop(
            columns=[col for col in columns_to_drop if col in df_with_label.columns]
        )

        # 初期データをコピーして保持
        df_extracted = pd.DataFrame()
        extracted_columns = set()

        df_pre = df_with_label.drop(columns=["date"])

        # 各エクストラクターに初期データを渡して実行
        for extractor in self.extractors:
            df_initial = df_pre.copy()
            if isinstance(extractor, SupervisedFeatureExtractorABC):
                df_temp = extractor.extract_features(df_initial, "label")
            else:
                df_temp = extractor.extract_features(df_initial.drop(columns=["label"]))

            for col in df_temp.columns:
                if col not in extracted_columns:
                    df_extracted[col] = df_temp[col]
                    extracted_columns.add(col)

        print(f"extracted feature columns({len(extracted_columns)})")
        # print(f"extracted feature columns({len(extracted_columns)}): {extracted_columns}")

        # 必要なカラムを追加
        df_extracted["date"] = df_with_label["date"]

        if symbol_value is not None:
            df_extracted["symbol"] = symbol_value

        # ラベル付きデータを準備
        if "label" in df_with_label.columns:
            df_extracted["label"] = df_with_label["label"]

        # カラムの順序を指定
        columns_order = ["date", "symbol"] + [
            col for col in df_extracted.columns if col not in ["date", "symbol"]
        ]
        df_extracted = df_extracted[columns_order]

        # データを保存
        self.extracted_f_w_l_d_manager.save_data(df_extracted, symbol)

        # print("Extractor stage completed successfully")
