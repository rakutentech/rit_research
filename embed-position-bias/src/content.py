import pandas as pd
from typing import Dict, List


class ContentModel:
    """Get contexts and predict expected reward.

    Args:
        content_model: A fitted scikit-learn-compatible estimator
        features: A list such that the estimator expects as input
        features appended with 'content_id'.
        content_ids: A list representing content_ids
        regression: A flag to indicate that the model is a regression model
        index_class: The index of the class to representing a reward.
    """

    def __init__(
        self, content_model, features: List, content_ids: List,
        regression: bool=False, index_class: int=-1
    ) -> None:
        if regression:
            raise NotImplementedError
        self._content_model = content_model
        self._index_class = index_class
        self._regression = regression
        self._content_ids = content_ids
        self._features = features

    def predict(self, context: pd.DataFrame) -> pd.DataFrame:
        context = pd.DataFrame(context, index=[0])[self._features]
        expected = {"contents": {}}
        for content_id in self._content_ids:
            context["content_id"] = content_id
            expected["contents"][content_id] = self._content_model.predict_proba(
                context
            )[0][self._index_class]
        return expected

    def predict_batch(self, contexts: pd.DataFrame) -> pd.DataFrame:
        contexts = contexts[self._features].copy(deep=False)
        expected = {"contents": pd.DataFrame(index=contexts.index)}
        for content_id in self._content_ids:
            contexts["content_id"] = content_id
            probs = self._content_model.predict_proba(contexts)[:, self._index_class]
            expected["contents"][content_id] = probs
        return expected

    def get_ids(self) -> Dict:
        ids = {"contents": self._content_ids}
        return ids
