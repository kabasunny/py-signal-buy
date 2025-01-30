from datetime import datetime, timedelta
import time
from models.ModelSaverLoader import ModelSaverLoader
from data.DataManager import DataManager
from result.ProtoSaverLoader import ProtoSaverLoader
from model_types import (
    model_types,
    cluster_model_types,
)  # 別ファイルで定義

from DataPreparationPipline import DataPreparationPipline
from ClusteringPipline import ClusteringPipline
from FeatureEngineeringPipline import FeatureEngineeringPipline
from ModelTrainingPipeline import (
    ModelTrainingPipeline,
)  # 過酷なトレーニングを専門とする
from ModelPredictionPipeline import (
    ModelPredictionPipeline,
)  # 実践シミュレーション用protofileを取り揃える


def main():
    current_date_str = datetime.now().strftime("%Y-%m-%d")

    trained_date_ago = 365 * 2  # トレーニング終了日 (2年前) 翌日以降実践
    # training_date_ago = trained_date_ago + 365 * 10 # トレーニング開始日（10年間の期間）

    # 以下はアプリケーション側の設定とする
    split_date = (datetime.now() - timedelta(days=trained_date_ago)).strftime(
        "%Y-%m-%d"
    )
    feature_period_days = 365 * 2  # 特徴量生成に必要なデータ期間、現在月足の期間に依存
    model_saver_loader = ModelSaverLoader(
        current_date_str, model_save_path="models/trained_models", model_file_ext="pkl"
    )

    feature_list_str = [
        "volume",
        "price",
        "past",
    ]

    cluster_ft_list_str = [
        "price_movement",
        "volume",
        "peak_trough",
        "fourier",
    ]

    extractors = ["PCA", "LDA", "ICA", "PCR"]

    selectors = [
        # "Tree",  # 決定木に基づく特徴量選択
        # "Lasso",  # Lasso回帰による特徴量選択
        # "Correlation",  # 相関に基づく特徴量選択
        # "MutualInformation",  # 相互情報量に基づく特徴量選択
        # "RFE",  # 再帰的特徴量削減
        # "VarianceThreshold",  # 分散閾値に基づく特徴量選択
        "SelectAll",  # 全特徴量を選択
    ]

    base_data_path = "data/stock_data"
    file_ext = "csv"  # "parquet"

    # ProtoSaverLoaderの初期化
    # proto_dir_path = "../go-optimal-stop/data/ml_stock_response"
    # proto_saver_loader = ProtoSaverLoader(proto_dir_path)

    data_manager_names = [
        "all_symbols",
        "formated_raw",
        "processed_raw",
        "labeled",
        "feature_for_cluster",
        "symbols_clustered_grp",
        "normalized_feature",
        "extracted_feature",
        "selected_feature",
        "training_and_test",
        "practical",
        "predictions",
    ]

    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(
            current_date_str, base_data_path, d_m_name, file_ext
        )

    data_preparation = DataPreparationPipline(
        feature_period_days,  # 特徴量生成に必要な日数
        data_managers,
    )

    clustering_pipline = ClusteringPipline(
        feature_period_days,  # 特徴量生成に必要な日数
        cluster_ft_list_str,
        cluster_model_types,
        data_managers,
    )

    feature_engineering = FeatureEngineeringPipline(
        feature_period_days,
        feature_list_str,
        extractors,
        selectors,
        data_managers,
    )

    training_pipeline = ModelTrainingPipeline(
        feature_period_days,  # 特徴量生成に必要な日数
        split_date,  # トレーニング最終日、翌日以降実践日
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
    )

    prediction_pipeline = ModelPredictionPipeline(
        feature_period_days,  # 特徴量生成に必要な日数
        split_date,
        model_types,
        feature_list_str,
        model_saver_loader,
        ProtoSaverLoader(),
        data_managers,
    )

    # 処理開始時間を記録 APIアクセスには1秒ラグを設けている
    start_time = time.time()

    # all_symbols = data_managers["all_symbols"].load_data("ticker_codes")

    # for _, row in all_symbols.iterrows():
    #     a_symbol = row["symbol"]
    #     data_preparation.process_symbol(a_symbol)  # 並列処理 可

    # 並列処理 不可
    # clustering_pipline.process()

    cluster_model_type = cluster_model_types[
        0
    ]  # 教師なし学習モデルを複数選択するかは検討

    clustered_files = data_managers["symbols_clustered_grp"].list_files_from_subdir(
        cluster_model_type
    )

    for i, clustered_file_num in enumerate(clustered_files, start=1):
        print(
            f"<< Now processing clustered_file_num {clustered_file_num} ,  {i} / {len(clustered_files)} >>"
        )
        clustered_symbols = data_managers[
            "symbols_clustered_grp"
        ].load_data_from_subdir(cluster_model_type, clustered_file_num)
        subdir = f"{cluster_model_type}/{clustered_file_num}"
        # クラスタが切り替わるタイミングで、教師ありモデルのインスタンスを新しく切り替える必要がある
        # for _, row in clustered_symbols.iterrows():
        #     one_symbol = row["symbol"]
            # feature_engineering.process_symbol(one_symbol)  # 並列処理 可
            # training_pipeline.process_symbol(
            #     one_symbol,
            #     subdir,
            # )  # 並列処理 不可
            # prediction_pipeline.process_symbol(
            #     one_symbol,
            #     subdir,
            # )  # 並列処理 可

        # シミュレーション用データ形式に変換
        prediction_pipeline.finish_prosess(
            clustered_symbols,
            subdir,
        )  # 並列処理 不可

    # 処理終了時間を記録
    end_time = time.time()

    # 処理時間を表示
    elapsed_time = end_time - start_time
    print(f"処理時間: {elapsed_time:.2f} 秒, All processing is complete")


if __name__ == "__main__":
    main()
