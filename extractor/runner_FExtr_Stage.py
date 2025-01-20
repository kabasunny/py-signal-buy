import sys
import os
import pandas as pd

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from extractor.FeatureExtractionStage import FeatureExtractionStage
from data.DataManager import DataManager
from extractor.ExtractorFactory import ExtractorFactory  # 新しく追加
from datetime import datetime

if __name__ == "__main__":
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    base_data_path = "data/stock_data/demo"
    file_ext = "csv"
    symbol = "1570"

    data_manager_names = [
        "labeled",
        "normalized_feature",
        "extracted_ft_with_label",
    ]

    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(
            current_date_str, base_data_path, d_m_name, file_ext
        )

    extractors = [
        "PCA",
        "LDA",
        "ICA",
        "PCR",
    ]

    # FeatureExtractionStage のインスタンスを作成し、実行
    FeatureExtractionStage(
        data_managers["labeled"],
        data_managers["normalized_feature"],
        data_managers["extracted_ft_with_label"],
        ExtractorFactory.create_extractors(extractors),
    ).run(symbol)
