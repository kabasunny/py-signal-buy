from result.ml_stock_service_pb2 import MLStockResponse
from result.print_ml_stock_response import print_ml_stock_response_summary
import os  # ディレクトリパスを扱うためにosモジュールをインポート


class ProtoSaverLoader:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def save_proto_response_to_file(self, proto_response, file_name):
        # ディレクトリが存在しない場合は作成
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)

        # 正しいファイルパスを作成
        full_file_path = os.path.join(self.directory_path, file_name)

        # プロトレスポンスをバイナリファイルに保存
        with open(full_file_path, "wb") as f:
            f.write(proto_response.SerializeToString())

        # print_ml_stock_response_summary(proto_response)

        return full_file_path

    def load_proto_response_from_file(self, file_name):
        full_file_path = os.path.join(self.directory_path, file_name)

        if not os.path.exists(full_file_path):
            raise ValueError(f"File at {full_file_path} does not exist")

        # プロトレスポンスをバイナリファイルからロード
        with open(full_file_path, "rb") as f:
            proto_response = MLStockResponse()
            proto_response.ParseFromString(f.read())
        return proto_response
