import pandas as pd
import numpy as np
from numpy.fft import fft
from .ClusteringFeatureCreatorABC import ClusteringFeatureCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class FourierClusteringFeatureCreator(ClusteringFeatureCreatorABC):
    @ArgsChecker((None, pd.DataFrame, pd.Timestamp), pd.DataFrame)
    def create_features(
        self, df: pd.DataFrame, ft_pred_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        フーリエ解析に基づいて特徴量を作成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            trade_start_date (pd.Timestamp): トレード開始日

        Returns:
            pd.DataFrame: 特徴量が追加されたデータフレーム
        """

        # 特徴量を格納する辞書を初期化
        feature_results = {}

        # `trade_start_date` 以降の終値データを抽出
        valid_prices = df[df["date"] >= ft_pred_start_date]["close"].values
        
        # データがない場合は空のデータフレームを返す
        if len(valid_prices) == 0:
            print("FourierClusteringFeatureCreator")
            return pd.DataFrame()

        # フーリエ特徴量を計算
        dominant_periods = self.fft_analysis(valid_prices)

        if len(dominant_periods) > 0:
            for i in range(3):  # 上位3つの周期を対象にする
                if i < len(dominant_periods):
                    feature_results[f"ff{i+1}_perd"] = dominant_periods[i]
                else:
                    feature_results[f"ff{i+1}_perd"] = np.nan
        else:
            for i in range(3):  # 上位3つの周期を対象にする
                feature_results[f"ff{i+1}_perd"] = np.nan

        # 結果をデータフレームに変換
        feature_df = pd.DataFrame([feature_results])

        return feature_df
    
    # フーリエ変換を行う関数をクラス内に移動
    def fft_analysis(self, prices):
        n = len(prices)
        fft_result = fft(prices)
        fft_amplitude = np.abs(fft_result)[: n // 2]
        fft_freq = np.fft.fftfreq(n, d=1)[: n // 2]

        # 周期計算
        non_zero_idx = fft_freq > 0
        if not any(non_zero_idx):
            return np.array([])  # すべて0の場合は空の配列を返す

        fft_period = 1 / fft_freq[non_zero_idx]
        fft_amplitude = fft_amplitude[non_zero_idx]

        # 振幅が大きい順に並べ替える
        sorted_indices = np.argsort(fft_amplitude)[::-1]
        sorted_periods = fft_period[sorted_indices]

        return sorted_periods
