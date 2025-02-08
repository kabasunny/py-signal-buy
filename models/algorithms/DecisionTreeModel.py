from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from typing import Any, Dict, Tuple
from models.BaseModelABC import BaseModelABC
from models.Evaluator import Evaluator
from decorators.ArgsChecker import ArgsChecker

class DecisionTreeModel(BaseModelABC):
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple["BaseModelABC", Dict[str, Any]]:
        self.model.fit(X_train, y_train)
        result = self.evaluate(X_test, y_test)
        return self, result

    @ArgsChecker((None, pd.DataFrame), pd.Series)
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X_test)
        return pd.Series(predictions)

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        result = Evaluator.evaluate_model(self.model, X_test, y_test)
        return result
