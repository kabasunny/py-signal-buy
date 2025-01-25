from clustering_features.ClusteringFeatureStage import ClusteringFeatureStage
from clustering_features.ClusteringFeatureCreatorFactory import (
    ClusteringFeatureCreatorFactory,
)
from clustering_models.ClusterGroupingStage import ClusterGroupingStage
import time


class ClusteringPipline:
    def __init__(
        self,
        feature_period_days,  # 特徴量生成に必要な日数
        cluster_ft_list_str,
        cluster_model_types,
        data_managers,
    ):
        self.feature_period_days = feature_period_days
        self.cluster_ft_list_str = cluster_ft_list_str
        self.cluster_model_types = cluster_model_types
        self.data_managers = data_managers

        # 各パイプラインをインスタンス変数として保持
        self.cluster_feature_stage = ClusteringFeatureStage(
            self.data_managers["processed_raw"],
            self.data_managers["feature_for_cluster"],
            self.feature_period_days,
            ClusteringFeatureCreatorFactory.create_feature_creators(
                self.cluster_ft_list_str
            ),
        )

        self.cluster_grouping_stage = ClusterGroupingStage(
            self.data_managers["feature_for_cluster"],
            self.data_managers["symbols_clustered_grp"],
            self.cluster_model_types,
        )

    def process(
        self,
    ):
        print(f"<< Now processing in {self.__class__.__name__} >>")
        try:
            stages = [
                ("ClusteringFeatureStage", self.cluster_feature_stage),
                ("ClusterGroupingStage", self.cluster_grouping_stage),
            ]

            for stage_name, stage in stages:
                start_time = time.time()
                stage.run()
                elapsed_time = time.time() - start_time
                print(f"処理時間: {elapsed_time:.4f} 秒, {stage_name} ")

        except Exception as e:
            print(f"{stage_name} 処理中にエラーが発生しました: {e}")
