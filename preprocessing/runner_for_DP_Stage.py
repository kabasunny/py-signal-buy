# opti-ml-py\preprocessing\runner.py
import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from preprocessing.DataPreprocessingStage import DataPreprocessingStage
from data.DataManager import DataManager
from datetime import datetime

if __name__ == "__main__":
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    before_period_days = 365 * 2  # 特徴量生成に必要なデータ期間
    base_data_path = "data/stock_data/demo"
    file_ext = "csv"  # "parquet"
    symbol = "1570"

    data_manager_names = [
        "formated_raw",
        "processed_raw",
    ]

    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(current_date_str, base_data_path, d_m_name, file_ext)

    # PreprocessPipeline のインスタンスを作成し、引数としてデータマネージャを渡す
    DataPreprocessingStage(data_managers["formated_raw"], data_managers["processed_raw"]).run(f"{symbol}")
