import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from features.FeatureEngineeringStage import FeatureEngineeringStage
from data.DataManager import DataManager
from AnalyzerFactory import AnalyzerFactory
from datetime import datetime

if __name__ == "__main__":
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    before_period_days = 365 * 2  # 特徴量生成に必要なデータ期間
    base_data_path = "data/stock_data/demo"
    file_ext = "csv"  # "parquet"
    symbol = "1570"

    data_manager_names = [
        "processed_raw",
        "normalized_feature",
    ]

    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(current_date_str, base_data_path, d_m_name, file_ext)
   
    # feature_list_str = ["peak_trough", "fourier", "volume", "price"]  # ノーマライズ時のエラーを回避済み
    
    feature_list_str = ["peak_trough", "fourier", "volume", "price", "past"]

    # FeaturePipeline のインスタンスを作成し、実行
    FeatureEngineeringStage(
            data_managers["processed_raw"],
            data_managers["normalized_feature"],
            before_period_days,
            AnalyzerFactory.create_analyzers(feature_list_str),
        ).run(f"{symbol}")
