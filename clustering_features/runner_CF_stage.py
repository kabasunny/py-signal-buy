import sys
import os
from datetime import datetime

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from clustering_features.ClusteringFeatureStage import ClusteringFeatureStage
from data.DataManager import DataManager
from clustering_features.ClusteringFeatureCreatorFactory import (
    ClusteringFeatureCreatorFactory,
)

if __name__ == "__main__":
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    before_period_days = 365 * 2  # 特徴量生成に必要なデータ期間
    base_data_path = "data/stock_data/demo"
    file_ext = "csv"  # "parquet"
    symbol = "1570"

    data_manager_names = [
        "processed_raw",
        "norm_ft_for_cluster",
    ]

    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(
            current_date_str, base_data_path, d_m_name, file_ext
        )

    feature_list_str = ["price_movement", "volume"]

    # ClusteringFeatureStage のインスタンスを作成し、実行
    ClusteringFeatureStage(
        data_managers["processed_raw"],
        data_managers["norm_ft_for_cluster"],
        before_period_days,
        ClusteringFeatureCreatorFactory.create_feature_creators(feature_list_str),
    ).run(symbol)
