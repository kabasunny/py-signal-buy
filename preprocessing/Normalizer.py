import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート

class Normalizer:
    @staticmethod
    @ArgsChecker((pd.DataFrame, list), pd.DataFrame)
    def normalize(df: pd.DataFrame, columns_to_normalize: list) -> pd.DataFrame:
        """
        指定された列のデータを0から1の範囲に正規化するメソッド。

        指定された列の数値データを0と1の範囲にスケーリングし、
        異なるスケールのデータを統一されたスケールに変換。

        Args:
            df (pd.DataFrame): 入力データフレーム
            columns_to_normalize (list): 正規化する列名のリスト

        Returns:
            pd.DataFrame: 正規化されたデータフレーム
        """
        # 存在しない列名を見つける
        missing_columns = [col for col in columns_to_normalize if col not in df.columns]
        
        if missing_columns:
            print(f"Skipping columns not found in DataFrame: {missing_columns}")
        
        # 存在する列名だけを選択
        columns_to_normalize = [col for col in columns_to_normalize if col in df.columns]
        
        if not columns_to_normalize:
            print("No columns to normalize found in the DataFrame.")
            return df
        
        scaler = MinMaxScaler()  # MinMaxScalerをインスタンス化
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])  # データを正規化
        return df
