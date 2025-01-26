import pandas as pd
import os
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class DataManager:
    @ArgsChecker((None, str, str, str, str), None)
    def __init__(
        self, date_str: str, base_path: str, data_manager_name: str, file_ext: str
    ):
        self.base_path = base_path
        self.file_ext = file_ext
        self.d_m_name = data_manager_name
        self.date_str = date_str

    def generate_path(self, filename: str) -> str:
        return f"{self.base_path}/{self.d_m_name}/{self.date_str}/{filename}.{self.file_ext}"

    def generate_subdir_path(self, subdirectory: str, filename: str) -> str:
        return f"{self.base_path}/{self.d_m_name}/{subdirectory}/{self.date_str}/{filename}.{self.file_ext}"

    @ArgsChecker((None, pd.DataFrame, str), None)
    def save_data(self, df: pd.DataFrame, filename: str):
        """データを保存するメソッド"""
        path = self.generate_path(filename)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if path.endswith(".csv"):
                df.to_csv(path, index=True)
            else:
                df.to_parquet(path, index=True)
            # print(f"データが {path} に保存されました")
        except Exception as e:
            print(f"{path}のデータ保存に失敗しました: {e}")

    @ArgsChecker((None, pd.DataFrame, str, str), None)
    def save_data_to_subdir(self, df: pd.DataFrame, subdirectory: str, filename: str):
        """サブディレクトリにデータを保存するメソッド"""
        path = self.generate_subdir_path(subdirectory, filename)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if path.endswith(".csv"):
                df.to_csv(path, index=True)
            else:
                df.to_parquet(path, index=True)
            # print(f"データが {path} に保存されました")
        except Exception as e:
            print(f"{path}のデータ保存に失敗しました: {e}")

    @ArgsChecker((None, str), pd.DataFrame)
    def load_data(self, filename: str) -> pd.DataFrame:
        """データをロードするメソッド"""
        dir_path = f"{self.base_path}/{self.d_m_name}/{self.date_str}/"
        if not os.path.exists(dir_path):
            parent_dir = f"{self.base_path}/{self.d_m_name}/"
            dir_list = sorted(os.listdir(parent_dir), reverse=True)
            for d in dir_list:
                potential_path = os.path.join(parent_dir, d)
                if os.path.isdir(potential_path):
                    dir_path = potential_path
                    break

        files = [
            f
            for f in os.listdir(dir_path)
            if f.startswith(filename) and f.endswith(self.file_ext)
        ]
        if not files:
            print(f"{filename}のデータファイルが存在しません。")
            return pd.DataFrame()

        latest_file = max(
            files, key=lambda x: os.path.getmtime(os.path.join(dir_path, x))
        )
        path = os.path.join(dir_path, latest_file)

        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, dtype={"symbol": str})  # dtype オプションを追加
            else:
                df = pd.read_parquet(path)
            df = df.loc[:, ~df.columns.str.contains("^Unnamed:")]
            # print(f"データが {path} からロードされました")
            return df
        except Exception as e:
            print(f"{path}のデータロードに失敗しました: {e}")
            return pd.DataFrame()

    @ArgsChecker((None, str, str), pd.DataFrame)
    def load_data_from_subdir(self, subdirectory: str, filename: str) -> pd.DataFrame:
        """サブディレクトリからデータをロードするメソッド"""
        dir_path = f"{self.base_path}/{self.d_m_name}/{subdirectory}/{self.date_str}/"
        if not os.path.exists(dir_path):
            parent_dir = f"{self.base_path}/{self.d_m_name}/{subdirectory}/"
            dir_list = sorted(os.listdir(parent_dir), reverse=True)
            for d in dir_list:
                potential_path = os.path.join(parent_dir, d)
                if os.path.isdir(potential_path):
                    dir_path = potential_path
                    break

        files = [
            f
            for f in os.listdir(dir_path)
            if f.startswith(filename) and f.endswith(self.file_ext)
        ]
        if not files:
            print(f"{filename}のデータファイルが存在しません。")
            return pd.DataFrame()

        latest_file = max(
            files, key=lambda x: os.path.getmtime(os.path.join(dir_path, x))
        )
        path = os.path.join(dir_path, latest_file)

        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, dtype={"symbol": str})
            else:
                df = pd.read_parquet(path)
            df = df.loc[:, ~df.columns.str.contains("^Unnamed:")]
            # print(f"データが {path} からロードされました")
            return df
        except Exception as e:
            print(f"{path}のデータロードに失敗しました: {e}")
            return pd.DataFrame()

    def list_files(self) -> list:
        """ディレクトリ内のファイル名（拡張子を除いた状態）をリストアップするメソッド"""
        dir_path = f"{self.base_path}/{self.d_m_name}/{self.date_str}/"
        if not os.path.exists(dir_path):
            return []

        file_names = [f for f in os.listdir(dir_path) if f.endswith(self.file_ext)]
        return [os.path.splitext(f)[0] for f in file_names]

    def list_files_from_subdir(self, subdirectory: str) -> list:
        """サブディレクトリ内のファイル名（拡張子を除いた状態）をリストアップするメソッド"""
        dir_path = f"{self.base_path}/{self.d_m_name}/{subdirectory}/{self.date_str}/"
        if not os.path.exists(dir_path):
            return []

        file_names = [f for f in os.listdir(dir_path) if f.endswith(self.file_ext)]
        return [os.path.splitext(f)[0] for f in file_names]
