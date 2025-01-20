from selector.PCAFeatureSelector import PCAFeatureSelector
from selector.CorrelationFeatureSelector import CorrelationFeatureSelector
from selector.LassoFeatureSelector import LassoFeatureSelector
from selector.TreeFeatureSelector import TreeFeatureSelector
from selector.MutualInformationSelector import MutualInformationSelector
from selector.RFESelector import RFESelector
from selector.VarianceThresholdSelector import VarianceThresholdSelector
from selector.SelectAllSelector import SelectAllSelector


class SelectorFactory:
    """
    SelectorFactoryクラスは、各種特徴量選択器を生成するためのファクトリクラス。
    """

    @staticmethod
    def create_selectors(selector_names):
        selectors = []
        for selector_name in selector_names:
            if selector_name == "Tree":
                selectors.append(TreeFeatureSelector(n_estimators=10, random_state=42))
            elif selector_name == "Lasso":
                selectors.append(LassoFeatureSelector(alpha=0.01))
            elif selector_name == "Correlation":
                selectors.append(CorrelationFeatureSelector(threshold=0.9))
            elif selector_name == "MutualInformation":
                selectors.append(MutualInformationSelector(k=10))
            elif selector_name == "RFE":
                selectors.append(RFESelector(n_features_to_select=10))
            elif selector_name == "VarianceThreshold":
                selectors.append(VarianceThresholdSelector(threshold=0.01))
            elif selector_name == "SelectAll":
                selectors.append(SelectAllSelector())
            else:
                raise ValueError(f"Unknown selector: {selector_name}")
            
        # print(selectors)
        return selectors
