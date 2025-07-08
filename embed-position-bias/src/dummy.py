from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from typing import Dict, List
import numpy as np
import pandas as pd


class DummyModel(BaseEstimator):
    """A classification model that fits a Bernoulli distribution
    to the data or uses a predefined distribution parameter if
    training on too few samples.
    Args:
        prob: a fixed probability which will be returned when
        sample size is insufficient.
        min_sample: a minimum number of samples to calculate prob.
    """

    def __init__(self, min_sample: int, prob: float=0.0) -> None:
        self.prob = prob
        self.min_sample = min_sample

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series=None):
        """Fits a Bernoulli distribution
        to the data or uses a fixed probabilty if sample size
        is insufficient.
        Args:
            X: A pandas DataFrame as data
            y: A pandas Series as target
            w: Sample weights for y
        """
        if len(X.index) >= self.min_sample:
            if sample_weight is not None:
                self._prob = np.sum(y * sample_weight) / sample_weight.sum()
            else:
                self._prob = y.sum() / len(y)
        else:
            self._prob = self.prob
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """Return a Bernoulli distribution by a trained or
        predefined probability.
        Args:
            X: A pandas DataFrame as data
        """
        if not hasattr(self, "_prob"):
            raise NotFittedError
        return np.array([1 - self._prob, self._prob] * len(X.index)).reshape(
            len(X.index), 2
        )
