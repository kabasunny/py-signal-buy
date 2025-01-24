from datetime import datetime, timedelta
from models.ModelSaverLoader import ModelSaverLoader
from data.DataManager import DataManager
from result.ProtoSaverLoader import ProtoSaverLoader
from symbols import symbols  # 別ファイルで定義
from model_types import model_types  # 別ファイルで定義
from DataPreparationPipline import DataPreparationPipline
from FeatureEngineeringPipline import FeatureEngineeringPipline
from ModelTrainingPipeline import (
    ModelTrainingPipeline,
)  # 過酷なトレーニングを専門とする
from ModelPredictionPipeline import (
    ModelPredictionPipeline,
)  # 実践シミュレーション用protofileを取り揃える
import time  # 追加


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

    base_data_path = "data/stock_data"
    file_ext = "csv"  # "parquet"

    data_manager_names = [
        "formated_raw",
        "processed_raw",
        "labeled",
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

    data_preparation = DataPreparationPipline(
        feature_period_days,  # 特徴量生成に必要な日数
        data_managers,
    )

    feature_engineering = FeatureEngineeringPipline(
        feature_period_days,
        feature_list_str,
        data_managers,
        extractors,
        selectors,
    )

    training_pipeline = ModelTrainingPipeline(
        feature_period_days,  # 特徴量生成に必要な日数
        split_date,  # トレーニング最終日、翌日以降実践日
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
    )

    # ProtoSaverLoaderの初期化
    proto_file_path = "../go-optimal-stop/data/ml_stock_response/latest_response.bin"
    proto_saver_loader = ProtoSaverLoader(proto_file_path)

    prediction_pipeline = ModelPredictionPipeline(
        feature_period_days,  # 特徴量生成に必要な日数
        split_date,
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
        proto_saver_loader,
    )

    # トレーニングと実データ用シンボルを取得
    symbols_copy = symbols.copy()
    # print(symbols)
    # print(symbols_copy)

    # 処理開始時間を記録
    start_time = time.time()

    while symbols_copy:
        symbol = symbols_copy.pop(0)
        data_preparation.process_symbol(symbol)
        feature_engineering.process_symbol(symbol)
        training_pipeline.process_symbol(symbol)
        prediction_pipeline.process_symbol(symbol)

    # シミュレーション用データ形式に変換
    prediction_pipeline.finish_prosess(symbols)

    # 処理終了時間を記録
    end_time = time.time()

    # 処理時間を表示
    elapsed_time = end_time - start_time
    print(f"処理時間: {elapsed_time:.2f} 秒, All processing is complete")


if __name__ == "__main__":
    main()
