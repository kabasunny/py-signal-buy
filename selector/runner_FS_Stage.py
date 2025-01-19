import sys
import os
import pandas as pd  # trade_start_date のために必要

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from selector.FeatureSelectionStage import FeatureSelectionStage
from data.DataManager import DataManager
from selector.SelectorFactory import SelectorFactory  # 新しく追加
from datetime import datetime

if __name__ == "__main__":
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    before_period_days = 365 * 2  # 特徴量生成に必要なデータ期間
    base_data_path = "data/stock_data/demo"
    file_ext = "csv"  # "parquet"
    symbol = "1570"

    data_manager_names = [
        "labeled",
        "normalized_feature",
        "selected_feature",
        "selected_ft_with_label",
    ]

    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(current_date_str, base_data_path, d_m_name, file_ext)

    selectors = [
        "Tree",
        "Lasso",
        "Correlation",
        "PCA",
        "SelectAll",
    ]
   
    # SelectorPipeline のインスタンスを作成し、実行
    FeatureSelectionStage(
            data_managers["labeled"],
            data_managers["normalized_feature"],
            data_managers["selected_feature"],
            data_managers["selected_ft_with_label"],
            SelectorFactory.create_selectors(selectors),
        ).run(f"{symbol}")

