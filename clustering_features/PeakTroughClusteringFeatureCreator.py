import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from .ClusteringFeatureCreatorABC import ClusteringFeatureCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PeakTroughClusteringFeatureCreator(ClusteringFeatureCreatorABC):
    @ArgsChecker((None, pd.DataFrame, pd.Timestamp), pd.DataFrame)
    def create_features(
        self, df: pd.DataFrame, ft_pred_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        ピーク・トラフ解析に基づいて特徴量を作成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            trade_start_date (pd.Timestamp): トレード開始日

        Returns:
            pd.DataFrame: 特徴量が追加されたデータフレーム
        """
        # ウィンドウサイズとリサンプルルールの定義
        window_sizes = [50, 30, 24]  # ここの月数が特徴量として最大の期間
        resample_rules = [None, "W", "ME"]

        # 特徴量名と設定
        feature_configs = {
            f"{size}{rule[0].lower() if rule else 'd'}": {
                "window_size": size,
                "resample": rule,
            }
            for size, rule in zip(window_sizes, resample_rules)
        }

        # データフレームを日付型に変換
        df["date"] = pd.to_datetime(df["date"])

        # リサンプリングを行うデータフレームを作成
        resampled_data = {}
        for feature_name, config in feature_configs.items():
            resample_rule = config["resample"]
            if resample_rule:
                resampled_data[feature_name] = (
                    df.set_index("date").resample(resample_rule).last()
                )

        # 特徴量を格納する辞書を初期化
        feature_results = {}

        for feature_name, config in feature_configs.items():
            window_size = config["window_size"]
            resample_rule = config["resample"]

            if resample_rule:
                data = resampled_data[feature_name]
                valid_data = data[data.index >= ft_pred_start_date]
            else:
                valid_data = df[df["date"] >= ft_pred_start_date]

            mean_intervals = []
            max_peaks = []
            min_troughs = []
            min_peaks = []  # 最小ピークを追加
            max_troughs = []  # 最大トラフを追加

            for idx in range(len(valid_data)):
                if resample_rule:
                    current_date = valid_data.index[idx]
                    try:
                        window_end_idx = pd.Index(data.index).get_indexer(
                            [current_date], method="ffill"
                        )[0]

                    except IndexError:
                        print(f"データが見つかりません: {current_date}")
                        continue
                    window_start_idx = max(0, window_end_idx - window_size)
                    # スライスの範囲をチェック
                    if (
                        window_start_idx >= 0
                        and window_start_idx < len(data)
                        and window_end_idx >= 0
                        and window_end_idx <= len(data)
                        and window_start_idx < window_end_idx
                    ):
                        window_data = data.iloc[window_start_idx:window_end_idx]
                    else:
                        continue

                else:
                    current_date = valid_data["date"].iloc[idx]
                    window_end_idx = df.index.get_loc(
                        df[df["date"] == current_date].iloc[0].name
                    )
                    window_start_idx = max(0, window_end_idx - window_size)
                    # スライスの範囲をチェック
                    if (
                        window_start_idx >= 0
                        and window_start_idx < len(df)
                        and window_end_idx >= 0
                        and window_end_idx <= len(df)
                        and window_start_idx < window_end_idx
                    ):
                        window_data = df.iloc[window_start_idx:window_end_idx]
                    else:
                        continue

                if window_data is not None:
                    prices = window_data["close"].values
                    peaks, _ = find_peaks(prices)
                    troughs, _ = find_peaks(-prices)

                    if len(peaks) > 0 and len(troughs) > 0:
                        if len(troughs) > 1:
                            intervals = np.diff(troughs)
                            mean_interval = np.mean(intervals)
                        else:
                            mean_interval = np.nan
                        mean_intervals.append(mean_interval)
                        max_peaks.append(np.max(prices[peaks]))
                        min_troughs.append(np.min(prices[troughs]))
                        min_peaks.append(np.min(prices[peaks]))  # 最小ピークを追加
                        max_troughs.append(np.max(prices[troughs]))  # 最大トラフを追加
            # 結果を辞書に格納
            if len(mean_intervals) > 0:
                feature_results[f"{feature_name}_me_intl"] = np.mean(mean_intervals)
                feature_results[f"{feature_name}_max_pk"] = (
                    np.max(max_peaks) if len(max_peaks) > 0 else np.nan
                )
                feature_results[f"{feature_name}_min_trh"] = (
                    np.min(min_troughs) if len(min_troughs) > 0 else np.nan
                )
                feature_results[f"{feature_name}_min_pk"] = (
                    np.min(min_peaks) if len(min_peaks) > 0 else np.nan
                )  # 最小ピークを追加
                feature_results[f"{feature_name}_max_trh"] = (
                    np.max(max_troughs) if len(max_troughs) > 0 else np.nan
                )  # 最大トラフを追加

                if (
                    len(max_peaks) > 0
                    and len(min_troughs) > 0
                    and np.min(min_troughs) != 0
                ):
                    feature_results[f"{feature_name}_pk_trh_ratio"] = (
                        np.max(max_peaks) - np.min(min_troughs)
                    ) / np.abs(np.min(min_troughs))
                else:
                    feature_results[f"{feature_name}_pk_trh_ratio"] = np.nan
            else:
                feature_results[f"{feature_name}_me_intl"] = np.nan
                feature_results[f"{feature_name}_max_pk"] = np.nan
                feature_results[f"{feature_name}_min_trh"] = np.nan
                feature_results[f"{feature_name}_min_pk"] = np.nan  # 最小ピークを追加
                feature_results[f"{feature_name}_max_trh"] = np.nan  # 最大トラフを追加
                feature_results[f"{feature_name}_pk_trh_ratio"] = np.nan

        # 結果をデータフレームに変換
        feature_df = pd.DataFrame([feature_results])

        return feature_df
