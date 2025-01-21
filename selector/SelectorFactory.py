from selector.algorithms.CorrelationFeatureSelector import CorrelationFeatureSelector
from selector.algorithms.LassoFeatureSelector import LassoFeatureSelector
from selector.algorithms.TreeFeatureSelector import TreeFeatureSelector
from selector.algorithms.MutualInformationSelector import MutualInformationSelector
from selector.algorithms.RFESelector import RFESelector
from selector.algorithms.VarianceThresholdSelector import VarianceThresholdSelector
from selector.algorithms.SelectAllSelector import SelectAllSelector


class SelectorFactory:
    """
    SelectorFactoryクラスは、各種特徴量選択器を生成するためのファクトリクラス。
    """

    @staticmethod
    def create_selectors(selector_names):
        selectors = []
        for selector_name in selector_names:
            if selector_name == "Tree":
                selectors.append(
                    TreeFeatureSelector(
                        n_estimators=100, max_features=0.1, random_state=42
                    )
                )  # n_estimatorsを増加し、max_featuresを減少
            elif selector_name == "Lasso":
                selectors.append(LassoFeatureSelector(alpha=0.005))  # alpha値を厳しく
            elif selector_name == "Correlation":
                selectors.append(
                    CorrelationFeatureSelector(threshold=0.75)
                )  # thresholdを厳しく
            elif selector_name == "MutualInformation":
                selectors.append(MutualInformationSelector(k=2))  # k値を厳しく
            elif selector_name == "RFE":
                selectors.append(
                    RFESelector(n_features_to_select=2)
                )  # n_features_to_selectを厳しく
            elif selector_name == "VarianceThreshold":
                selectors.append(
                    VarianceThresholdSelector(threshold=1)
                )  # thresholdを厳しく
            elif selector_name == "SelectAll":
                selectors.append(SelectAllSelector())
            else:
                raise ValueError(f"Unknown selector: {selector_name}")

        return selectors
