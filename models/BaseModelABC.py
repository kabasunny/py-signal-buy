from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import pandas as pd

class BaseModelABC(ABC):

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple["BaseModelABC", Dict[str, Any]]:
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> Any:
        pass

    @abstractmethod
    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        pass
