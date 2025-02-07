import sys
import os
from datetime import datetime

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from clustering_models.ClusterGroupingStage import ClusterGroupingStage
from data.DataManager import DataManager

if __name__ == "__main__":
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    base_data_path = "data/stock_data/demo"
    file_ext = "csv"  # "parquet"

    data_manager_names = [
        "feature_for_cluster",
        "symbols_clusted_grp",
    ]

    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(
            current_date_str, base_data_path, d_m_name, file_ext
        )

    cluster_model_types = [
        "kmeans",
        "dbscan",
        "hierarchical",
        "mean_shift",
        "gmm",
        "spectral",
    ]  # 使用するモデルのリスト

    # ClusterGroupingStage のインスタンスを作成し、実行
    cluster_grouping_stage = ClusterGroupingStage(
        data_managers["feature_for_cluster"],
        data_managers["symbols_clusted_grp"],
        cluster_model_types,
    )
    cluster_grouping_stage.run()
