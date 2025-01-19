from datetime import datetime

from models.ModelSaverLoader import ModelSaverLoader
from data.DataManager import DataManager
from proto_conversion.ProtoSaverLoader import ProtoSaverLoader
from symbols import get_train_and_real_data_symbols  # 別ファイルで定義
from model_types import model_types  # 別ファイルで定義

from TrainAutomatedPipeline import TrainAutomatedPipeline  # 過酷なトレーニングを専門とする
from RealDataAutomatedPipeline import RealDataAutomatedPipeline  # 実践シミュレーション用protofileを取り揃える

def main():
    current_date_str = datetime.now().strftime("%Y-%m-%d")

    before_period_days = 365 * 2  # 特徴量生成に必要なデータ期間

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
        "real_predictions",
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

    train_pipeline = TrainAutomatedPipeline(
        before_period_days,
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
        selectors,
    )

    # ProtoSaverLoaderの初期化
    proto_file_path = "../go-optimal-stop/data/ml_stock_response/latest_response.bin"
    proto_saver_loader = ProtoSaverLoader(proto_file_path)

    real_data_pipeline = RealDataAutomatedPipeline(
        before_period_days,
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
        selectors,
        proto_saver_loader
    )

    # トレーニングと実データ用シンボルを取得
    train_symbols, real_data_symbols = get_train_and_real_data_symbols(train_ratio=0.9)
    r_d_symbols_copy = real_data_symbols.copy()
    print(f"train_symbols : {train_symbols}")
    print(f"real_data_symbols : {real_data_symbols}")

    # トレーニングを行う
    while train_symbols:
        train_symbol = train_symbols.pop(0)
        train_pipeline.process_symbol(train_symbol)

    # 予測結果を作成
    while real_data_symbols:
        real_data_symbol = real_data_symbols.pop(0)
        real_data_pipeline.process_symbol(real_data_symbol)
    
    # シミュレーション用データ形式に変換
    real_data_pipeline.finish_prosess(r_d_symbols_copy)

if __name__ == "__main__":
    main()
