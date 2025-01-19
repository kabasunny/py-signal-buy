from datetime import datetime, timedelta

from models.ModelSaverLoader import ModelSaverLoader
from data.DataManager import DataManager
from result.ProtoSaverLoader import ProtoSaverLoader
from symbols import symbols  # 別ファイルで定義
from model_types import model_types  # 別ファイルで定義
from DataPreparationPipline import DataPreparationPipline
from ModelTrainingPipeline import ModelTrainingPipeline  # 過酷なトレーニングを専門とする
from ModelPredictionPipeline import ModelPredictionPipeline  # 実践シミュレーション用protofileを取り揃える

def main():
    current_date_str = datetime.now().strftime("%Y-%m-%d")

    before_period_days = 365 * 2  # 特徴量生成に必要なデータ期間# 現在の日付から2年前の日付を計算
    trained_date_ago = 365 * 2 # トレーニング終了日 (2年前) 翌日以降実践
    # training_date_ago = trained_date_ago + 365 * 10 # トレーニング開始日（10年間の期間） 
    split_date = (datetime.now() - timedelta(days=trained_date_ago)).strftime("%Y-%m-%d")

    model_saver_loader = ModelSaverLoader(
        current_date_str, model_save_path="models/trained_models", model_file_ext="pkl"
    )

    feature_list_str = ["peak_trough", "fourier", "volume", "price", "past"]

    base_data_path = "data/stock_data"
    file_ext = "csv"  # "parquet"

    data_manager_names = [
        "formated_raw",
        "processed_raw",
        "labeled",
        "normalized_feature",
        "selected_feature",
        "selected_ft_with_label",
        "training_and_test",
        "practical",
        "predictions",
    ]

    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(current_date_str, base_data_path, d_m_name, file_ext)

    selectors = [
        # "Tree",
        # "Lasso",
        # "Correlation",
        "PCA",
        "SelectAll",
    ]

    data_preparation = DataPreparationPipline(
        before_period_days,  # 特徴量生成に必要な日数
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
        selectors  # 新しい引数を追加
    )

    training_pipeline = ModelTrainingPipeline(
        before_period_days,  # 特徴量生成に必要な日数
        split_date, # トレーニング最終日、翌日以降実践日
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
        selectors  # 新しい引数を追加
    )

    # ProtoSaverLoaderの初期化
    proto_file_path = "../go-optimal-stop/data/ml_stock_response/latest_response.bin"
    proto_saver_loader = ProtoSaverLoader(proto_file_path)

    prediction_pipeline = ModelPredictionPipeline(
        before_period_days,  # 特徴量生成に必要な日数
        split_date,
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
        selectors,
        proto_saver_loader
    )

    # トレーニングと実データ用シンボルを取得
    symbols_copy = symbols.copy()

    while symbols_copy:
        symbol = symbols_copy.pop(0)
        data_preparation.process_symbol(symbol)
        training_pipeline.process_symbol(symbol)
        prediction_pipeline.process_symbol(symbol)
    
    # シミュレーション用データ形式に変換
    prediction_pipeline.finish_prosess(symbols)

if __name__ == "__main__":
    main()
