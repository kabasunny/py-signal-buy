import sys
import os
from datetime import datetime, timedelta

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from result.print_ml_stock_response import print_ml_stock_response_summary
from data.DataManager import DataManager
from datetime import datetime
from result.ResultSavingStage import ResultSavingStage
from result.ProtoSaverLoader import ProtoSaverLoader
# from symbols import symbols  # 別ファイルで定義
from model_types import model_types  # 別ファイルで定義

if __name__ == "__main__":
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    trained_date_ago = 365 * 2 # トレーニング終了日 (2年前) 翌日以降実践
    base_data_path = "data/stock_data/demo"
    file_ext = "csv"  # "parquet"

    # 現在の日付から2年前の日付を計算
    split_date = (datetime.now() - timedelta(days=trained_date_ago)).strftime("%Y-%m-%d")

     # データマネージャのインスタンスを作成
    data_manager_names = [
        "formated_raw",
        "predictions",
    ]
    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(current_date_str, base_data_path, d_m_name, file_ext)


    file_path = "../go-optimal-stop/data/ml_stock_response/demo/latest_response.bin"

    # ProtoSaverLoaderの初期化
    proto_saver_loader = ProtoSaverLoader(file_path)

    symbols = ["1570",] # ここはリストで渡す

    # ProtoConvertPipelineの初期化と実行
    ResultSavingStage(
        data_managers["formated_raw"],
        data_managers["predictions"],
        proto_saver_loader,
        model_types,
        split_date,
        ).run(symbols) # リストを受けるため他のパイプラインと異なる
    print("Proto conversion and saving completed.")


    # 保存したプロトコルバッファーの読み込み
    loaded_proto_response = proto_saver_loader.load_proto_response_from_file()
    print_ml_stock_response_summary(loaded_proto_response)

# MLStockResponse (summary):
# Symbol: 1570
#   Daily Data:
#     Date: 2012-04-09, Open: 4351.0, Close: 4351.0
#     Date: 2012-04-10, Open: 4345.35986328125, Close: 4345.35986328125
#     ...
#     Date: 2025-01-09, Open: 28050.0, Close: 27740.0
#     Date: 2025-01-10, Open: 27295.0, Close: 27175.0
#   Signals: ['2014-05-19', '2014-05-20'] ... ['2024-12-19', '2024-12-20']
#   Model Predictions:
#     AdaBoost: ['2014-10-16', '2014-10-20'] ... ['2024-09-12', '2024-09-18']
#     KNeighbors: ['2014-10-22', '2014-10-24'] ... ['2024-08-02', '2024-09-12']
#     XGBoost: ['2014-10-16', '2014-10-20'] ... ['2024-09-18', '2024-09-19']
#     SVM: [] ... []
#     LightGBM: ['2014-10-20', '2014-10-21'] ... ['2024-09-13', '2024-09-18']
#     RandomForest: ['2014-10-16', '2014-10-24'] ... ['2024-09-13', '2024-09-18']
#     CatBoost: ['2014-10-16', '2014-10-20'] ... ['2024-09-18', '2024-09-19']
#     LogisticRegression: ['2015-09-03', '2015-09-08'] ... ['2024-08-06', '2024-08-09']
