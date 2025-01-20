import sys
import os
import pandas as pd  # trade_start_date のために必要

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from labeling.LabelCreationStage import LabelCreateStage
from data.DataManager import DataManager
from labeling.TroughLabelCreator import TroughLabelCreator
from datetime import datetime

if __name__ == "__main__":
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    before_period_days = 365 * 2  # 特徴量生成に必要なデータ期間
    base_data_path = "data/stock_data/demo"
    file_ext = "csv"  # "parquet"
    symbol = "1570"

    data_manager_names = [
        "formated_raw",
        "labeled",
    ]

    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(
            current_date_str, base_data_path, d_m_name, file_ext
        )

    # LabelCreationPipeline のインスタンスを作成し、実行
    LabelCreateStage(
        data_managers["formated_raw"],
        data_managers["labeled"],
        before_period_days,
        TroughLabelCreator(),
    ).run(f"{symbol}")
