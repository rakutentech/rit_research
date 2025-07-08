from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from copy import deepcopy
from typing import Dict, List
import pandas as pd
import numpy as np


class ModelSwitcher(BaseEstimator):
    """A class that handles multiple classification models based on
    switch argument. It divides training pandas dataframe by the
    value of switch column. For each divided sub-dataframe we fit
    model and predict probabilities transparently. If ValueError was
    raised during fitting, it will use DummyModel instead.
    Args:
        estimator: a scikit learn estimator for classification
        switch: a string which indicates a name of column to divide
    """

    def __init__(self, estimator, switch):
        self.estimator = estimator
        self.switch = switch

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Divides samples based on switch column and fits models.
        Args:
            X: A pandas DataFrame as data
            y: A pandas Series as target

        Keyword Arg:
            **kwargs: Arbitrary keyword arguments.
        """
        self._categories = X[self.switch].unique()
        self._estimators = {}
        for cat in self._categories:
            mask = (X[self.switch] == cat).values
            X_cat = X[mask].drop(columns=self.switch)
            kwargs_cat = {key: val[mask] for key, val in kwargs.items()}
            self._estimators[cat] = deepcopy(self.estimator).fit(
                X_cat, y[mask], **kwargs_cat
            )

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict probabilities of targets from data
        Args:
            X: A pandas DataFrame as data
        """
        if not hasattr(self, "_estimators"):
            raise NotFittedError
        X = X.reset_index(drop=True)
        res = np.zeros((X.shape[0], 2))
        for cat in X[self.switch].unique():
            mask = (X[self.switch] == cat).values
            X_cat = X[mask].drop(columns=self.switch)
            est_cat = self._estimators.get(cat)
            if est_cat is not None:
                proba = est_cat.predict_proba(X_cat)
                res[X_cat.index] = proba

        return res

    def get_estimators(self):
        """Get estimators as dict"""
        if not hasattr(self, "_estimators"):
            raise NotFittedError
        return self._estimators
