# opti-ml-py\models\runner.py
from datetime import datetime, timedelta
import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from data.DataManager import DataManager
from models.ModelTrainStage import ModelTrainStage
from models.ModelSaverLoader import ModelSaverLoader
from model_types import model_types  # 別ファイルで定義


def runner():
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    base_data_path = "data/stock_data/demo"
    file_ext = "csv"  # "parquet"
    symbol = "1570"

    data_manager_names = [
        "training_and_test",
    ]

    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(current_date_str, base_data_path, d_m_name, file_ext)

    model_save_path = "models/trained_models/demo"
    model_file_ext = "pkl"
    # モデルセーブローダーのインスタンスを作成
    model_saver_loader = ModelSaverLoader(current_date_str, model_save_path, model_file_ext)

    # モデルパイプラインの作成と実行
    ModelTrainStage(
            data_managers["training_and_test"],
            model_saver_loader,
            model_types,
            ).run(symbol)


if __name__ == "__main__":
    runner()
