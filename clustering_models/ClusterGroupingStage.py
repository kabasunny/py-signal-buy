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
        norm_ft_for_cluster: DataManager,
        symbols_clusted_grp: DataManager,
        model_types: list,
        days: int,
    ):
        self.norm_ft_for_cluster = norm_ft_for_cluster
        self.symbols_clusted_grp = symbols_clusted_grp
        self.model_types = model_types
        self.days = days

    def run(self):
        """
        クラスタリングを実行し、クラスタラベルを付与するメソッド。
        """
        # 今日の日付と指定日数前の日付を計算
        today = datetime.now()
        start_date = today - timedelta(days=self.days)

        # ディレクトリ内のすべてのCSVファイルを取得
        files = self.norm_ft_for_cluster.list_files()

        all_data = []

        for file in files:
            symbol = file
            # 正規化されたデータをロード
            df = self.norm_ft_for_cluster.load_data(symbol)

            # データフレームが空でないことを確認
            if df.empty:
                print(f"{symbol} をスキップします")
                continue

            # 指定日数内のデータにフィルタリング
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start_date) & (df["date"] <= today)]

            # シンボル列が既に存在するかを確認
            if "symbol" not in df.columns:
                # シンボル列を追加
                df["symbol"] = symbol

            # データを統合
            all_data.append(df)

        # データフレームに変換
        all_data_df = pd.concat(all_data)

        # クラスタリング用の特徴量を選択
        feature_columns = [
            "daily_return",
            "volatility",
            "volume_change_rate",
            "volume_moving_average",
        ]
        feature_data = all_data_df[feature_columns]

        for model_type in self.model_types:
            # モデルのインスタンスを生成
            model = ClusteringModelFactory.create_model(model_type)

            # クラスタリングを実行
            clusters = model.fit_predict(feature_data)
            all_data_df["cluster"] = clusters

            # クラスタごとにシンボルリストを生成
            cluster_symbols = all_data_df[["symbol", "cluster"]].drop_duplicates()
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
