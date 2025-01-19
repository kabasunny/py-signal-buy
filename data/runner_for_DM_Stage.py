import sys
import os
from datetime import datetime, timedelta

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.DataManager import DataManager
from data.DataForModelStage import DataForModelStage

if __name__ == "__main__":
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    before_period_days = 365 * 2  # 特徴量生成に必要なデータ期間
    base_data_path = "data/stock_data/demo"
    file_ext = "csv"  # "parquet"
    symbol = "1570"

    # 現在の日付から2年前の日付を計算
    split_date = (datetime.now() - timedelta(days=before_period_days)).strftime("%Y-%m-%d")

    data_manager_names = [
        "selected_ft_with_label",
        "training_and_test",
        "practical",
    ]

    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(current_date_str, base_data_path, d_m_name, file_ext)

    # PrepareDataForModelPipeline のインスタンスを作成し、実行
    DataForModelStage(
        data_managers["selected_ft_with_label"],
        data_managers["training_and_test"],
        data_managers["practical"],
        split_date
    ).run(f"{symbol}")
