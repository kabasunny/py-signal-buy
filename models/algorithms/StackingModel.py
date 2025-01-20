from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Any, Tuple
from models.BaseModelABC import BaseModelABC
from models.Evaluator import Evaluator
from decorators.ArgsChecker import ArgsChecker

class StackingModel(BaseModelABC):
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = StackingClassifier(estimators=[
            ('lr', LogisticRegression(solver='lbfgs', max_iter=2000)),
            ('dt', DecisionTreeClassifier()),
            ('rf', RandomForestClassifier())
        ], final_estimator=LogisticRegression(solver='lbfgs', max_iter=2000))

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple["BaseModelABC", Tuple[float, float, float, float]]:
        # データのスケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # ndarray から DataFrame に変換
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        self.model.fit(X_train_scaled_df, y_train)
        result = self.evaluate(X_test_scaled_df, y_test)
        return self, result

    @ArgsChecker((None, pd.DataFrame), pd.Series)
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        predictions = self.model.predict(X_test_scaled_df)
        return pd.Series(predictions)

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[float, float, float, float]:
        return Evaluator.evaluate_model(self, X_test, y_test)
