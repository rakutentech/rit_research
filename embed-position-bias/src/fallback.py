from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
import pandas as pd


class FallbackModel(BaseEstimator):
    """A model which takes two scikit-learn models as
    estimator and fallback. First this model tries to
    fit by estimator but if failed it uses fallback.
    Args:
        estimator: A scikit-learn estimator to fit
        usually
        fallback: A scikit-learn estimator to fit
        when this class catched a ValueError.
    """

    def __init__(self, estimator, fallback):
        self.estimator = estimator
        self.fallback = fallback

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Tries to fit estimator, but if it catched
        ValueError it fits fallback instead.
        Args:
            X: A pandas DataFrame as data
            y: A pandas Series as target
            **kwargs: Arbitrary keyword arguments.
        """
        try:
            self._model = self.estimator.fit(X, y, **kwargs)
            self.fellback_ = False
        except ValueError:
            self._model = self.fallback.fit(X, y, **kwargs)
            self.fellback_ = True
        return self

    def predict_proba(self, X: pd.DataFrame):
        """Predict probs of targets from data
        Args:
            X: A pandas DataFrame as data
        """
        if not hasattr(self, "_model"):
            raise NotFittedError
        return self._model.predict_proba(X)

    def get_model(self):
        """Get a evaluated model, which is either
        estimator or fallback.
        """
        if not hasattr(self, "_model"):
            raise NotFittedError
        return self._model
