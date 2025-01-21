import pandas as pd
from selector.SupervisedFeatureSelectorABC import SupervisedFeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker
from data.DataManager import DataManager


class FeatureSelectionStage:
    @ArgsChecker((None, DataManager, DataManager, DataManager, DataManager, list), None)
    def __init__(
        self,
        label_data_manager: DataManager,
        normalized_f_d_manager: DataManager,
        extracted_f_w_l_d_manager: DataManager,
        selected_f_w_l_d_manager: DataManager,
        selectors: list,
    ):
        self.label_data_manager = label_data_manager
        self.normalized_f_d_manager = normalized_f_d_manager
        self.extracted_f_w_l_d_manager = extracted_f_w_l_d_manager
        self.selected_f_w_l_d_manager = selected_f_w_l_d_manager
        self.selectors = selectors

    @ArgsChecker((None, str), None)
    def run(self, symbol) -> None:
        """
        特徴量選択を一連の流れで実行するメソッド

        Args:
            symbol (str): 対象のシンボル名
        """
        # 正規化済みデータをロード
        df_normalized = self.normalized_f_d_manager.load_data(symbol)
        # データフレームが空でないことを確認
        if df_normalized.empty:
            print(f" {symbol} をスキップします")
            return

        # date カラムを Timestamp 型に変換
        if "date" in df_normalized.columns:
            df_normalized["date"] = pd.to_datetime(
                df_normalized["date"], errors="coerce"
            )

        # 抽出された特徴量データをロード
        df_extracted = self.extracted_f_w_l_d_manager.load_data(symbol)
        # データフレームが空でないことを確認
        if df_extracted.empty:
            print(f" {symbol} をスキップします")
            return

        # date カラムを Timestamp 型に変換
        if "date" in df_extracted.columns:
            df_extracted["date"] = pd.to_datetime(df_extracted["date"], errors="coerce")

        # 抽出された特徴量データと正規化済みデータをマージ
        df_combined = df_normalized.merge(df_extracted, on="date")

        # ラベルデータを読み込んでマージ
        target_df = self.label_data_manager.load_data(symbol)
        # データフレームが空でないことを確認
        if target_df.empty:
            print(f" {symbol} をスキップします")
            return

        # date カラムを Timestamp 型に変換
        if "date" in target_df.columns:
            target_df["date"] = pd.to_datetime(target_df["date"], errors="coerce")

        df_with_label = df_combined.merge(
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
        columns_to_drop = ["symbol", "open", "high", "low", "close", "volume"]
        df_with_label = df_with_label.drop(
            columns=[col for col in columns_to_drop if col in df_with_label.columns]
        )

        # 初期データをコピーして保持
        df_selected = pd.DataFrame()
        selected_columns = set()  # 選択された特徴量を保持するためのセット

        df_pre = df_with_label.drop(columns=["date"])  # date列を削除

        # 各セレクターに初期データを渡して実行
        for selector in self.selectors:
            df_initial = df_pre.copy()
            # print(f"Running selector: {selector.__class__.__name__}")
            # print(f"Columns in df_initial: {df_initial.columns}")

            if "label" not in df_initial.columns:
                raise KeyError("The 'label' column is missing from the dataframe.")

            if isinstance(selector, SupervisedFeatureSelectorABC):
                # 特徴量選択（ラベル付き）
                df_temp = selector.select_features(df_initial, "label")
            else:
                # 特徴量選択（ラベルなし）
                df_temp = selector.select_features(df_initial.drop(columns=["label"]))

            # 重複する特徴量を追加しないように選択
            for col in df_temp.columns:
                if col not in selected_columns:
                    df_selected[col] = df_temp[col]
                    selected_columns.add(col)

        # 除外されたカラムを表示
        excluded_columns = (
            set(df_pre.drop(columns=["label"]).columns) - selected_columns
        )
        print(f"selected feature columns({len(selected_columns)})")
        print(f"Excluded feature columns({len(excluded_columns)})")
        # print(f"selected feature columns({len(selected_columns)}): {selected_columns}")
        # print(f"Excluded feature columns({len(excluded_columns)}): {excluded_columns}")

        # 必要なカラムを追加
        df_selected["date"] = df_with_label["date"]

        if symbol_value is not None:
            df_selected["symbol"] = symbol_value

        # `label` カラムを削除して保存する
        if "label" in df_selected.columns:
            df_selected = df_selected.drop(columns=["label"])

        # カラムの順序を指定
        columns_order = ["date", "symbol"] + [
            col for col in df_selected.columns if col not in ["date", "symbol"]
        ]
        df_selected = df_selected[columns_order]

        # データを保存
        self.selected_f_w_l_d_manager.save_data(df_selected, symbol)

        # print("Selector pipeline completed successfully")
