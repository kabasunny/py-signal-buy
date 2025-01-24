import pandas as pd
from datetime import datetime, timedelta
from data.DataManager import DataManager
from clustering_models.ClusteringModelFactory import ClusteringModelFactory
from decorators.ArgsChecker import ArgsChecker


class ClusterGroupingStage:
    """
    クラスタリングのためのステージクラス。
    """

    @ArgsChecker((None, DataManager, DataManager, list, int), None)
    def __init__(
        self,
        feature_for_cluster: DataManager,
        symbols_clusted_grp: DataManager,
        model_types: list,
        days: int,
    ):
        self.feature_for_cluster = feature_for_cluster
        self.symbols_clusted_grp = symbols_clusted_grp
        self.model_types = model_types
        self.days = days

    def run(self):
        """
        クラスタリングを実行し、クラスタラベルを付与するメソッド。
        """

        # ディレクトリ内のCSVファイルを取得（1つのみ想定）
        files = self.feature_for_cluster.list_files()
        if not files:
            print("クラスタリング用のCSVファイルが存在しません")
            return

        file = files[0]

        # 正規化されたデータをロード
        df = self.feature_for_cluster.load_data(file)

        # データフレームが空でないことを確認
        if df.empty:
            print(f"{file} にクラスタリング用特徴データがありません")
            return

        # シンボル列が既に存在するかを確認
        if "symbol" not in df.columns:
            # シンボル列を追加
            df["symbol"] = file

        # クラスタリング用の特徴量を自動的に選択（symbol列以外のすべて）
        feature_columns = df.columns.difference(["symbol", "date"]).tolist()
        feature_data = df[feature_columns]

        for model_type in self.model_types:
            # モデルのインスタンスを生成
            model = ClusteringModelFactory.create_model(model_type)

            # クラスタリングを実行
            clusters = model.fit_predict(feature_data)
            df["cluster"] = clusters

            # クラスタごとにシンボルリストを生成
            cluster_symbols = df[["symbol", "cluster"]].drop_duplicates()
            cluster_groups = (
                cluster_symbols.groupby("cluster")["symbol"].apply(list).reset_index()
            )

            # クラスタごとにシンボルリストを保存
            for _, row in cluster_groups.iterrows():
                cluster_label = row["cluster"]
                symbols = row["symbol"]
                output_df = pd.DataFrame({"symbol": symbols})

                # データを保存
                self.symbols_clusted_grp.save_data(
                    output_df, f"{model_type}_cluster_{cluster_label}"
                )

        print(f"Clustering completed for all models: {self.model_types}")
