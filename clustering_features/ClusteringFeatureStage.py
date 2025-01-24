import pandas as pd
import numpy as np
from preprocessing.Normalizer import Normalizer
from decorators.ArgsChecker import ArgsChecker
from data.DataManager import DataManager


class ClusteringFeatureStage:
    """
    特徴量生成の全体の流れを管理し、各特徴量作成クラスを呼び出すクラス。
    """

    @ArgsChecker((None, DataManager, DataManager, int, list), None)
    def __init__(
        self,
        processed_data_manager: DataManager,
        norm_ft_for_cluster: DataManager,
        feature_period_days: int,
        feature_creators: list,
    ):
        self.normalizer = Normalizer()
        self.processed_data_manager = processed_data_manager
        self.norm_ft_for_cluster = norm_ft_for_cluster
        self.feature_creators = feature_creators
        self.feature_period_days = feature_period_days

    @ArgsChecker((None,), None)
    def run(self):
        """
        特徴量作成と選択を一連の流れで実行するメソッド
        """
        symbols = self.processed_data_manager.list_files()
        all_features = []

        for symbol in symbols:
            df = self.processed_data_manager.load_data(symbol)
            if df.empty:
                print(f"{symbol} をスキップします")
                continue

            first_date = pd.to_datetime(df["date"].iloc[0])
            feature_start_date = first_date + pd.DateOffset(
                days=self.feature_period_days
            )
            df["date"] = pd.to_datetime(df["date"])

            feature_dfs = []
            for creator in self.feature_creators:
                stats = creator.create_features(df, feature_start_date)
                feature_dfs.append(stats)

            if feature_dfs:
                df_stats = pd.concat(feature_dfs, axis=1)
                df_stats["symbol"] = symbol
                df_stats.replace(
                    [float("inf"), -float("inf")], float("nan"), inplace=True
                )
                df_stats.fillna(0, inplace=True)
                all_features.append(df_stats)

        if all_features:
            final_df = pd.concat(all_features, ignore_index=True)
            cols = ["symbol"] + [col for col in final_df if col != "symbol"]
            final_df = final_df[cols]

            # 正規化前のデータを保存
            # self.norm_ft_for_cluster.save_data(final_df, "all_feature_before_norm")

            columns_to_normalize = final_df.select_dtypes(
                include=np.number
            ).columns.tolist()
            df_normalized = self.normalizer.normalize(final_df, columns_to_normalize)
            self.norm_ft_for_cluster.save_data(df_normalized, "all_features")

        print("Feature creation pipeline completed successfully")
