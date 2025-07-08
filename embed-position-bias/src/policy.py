


import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from typing import Dict, List
from obp.policy.pbm.dummy import DummyModel
from obp.policy.pbm.fallback import FallbackModel
from obp.policy.pbm.model_switcher import (
    ModelSwitcher,
)
from obp.policy.pbm.slot_optimizer import (
    SlotOptimizer,
)


class LogisticEpsilonGreedy:
    def __init__(self, action_spec: Dict, policy_state_spec: Dict={}, time_step_spec: Dict={}) -> None:
        """
        Base policy for multi-slot optimization.

        This policy is a conbination of arbitary regression model and
        epsilon greedy policy.
        For each arm, we'll have a regression model.
        ModelSwitcher will divide data and train transperatenly.
        Regression model will require some amount of data to have an
        accurate prediction, and conduct CV.
        Therefore, we introducted fallback system. When the data is
        insufficient, dummy model will be used insteadly.
        Prediction of dummy model is just a ratio of positive samples.

        Currently it only supports when num of banners = num of positions.

        Args:
            action_spec: a dict which stores config of arms
                arms: list
                    name of arms
                slots: list
                    name of slots
            policy_state_spec: a dict which stores config of policy
                epsilon: int, default 1
                    init of epsilon value
                epsilon_decay: float, default 0.9
                    decay of epsilon at each round
                epsilon_min: float, default 0.01
                    min of epsilon. The minimum ratio of exploration
                pesonalized: bool
                    If ture, we conduct regression, if not we use dummy model
                optimize_slot: bool
                    If true, we estimate position bias by EM-algorithm.
                    If not, use pre_trained position bias
        Attributes:
            arms: list of arms as strings
            num_arms: num of arms as integer
            _success: list of success counts as int
            _t: a value which discribes total steps so far
        """

        self._success = None
        self._t = None
        self._time_step_spec = time_step_spec
        self._action_spec = action_spec
        self.arms = sorted(action_spec["arms"])
        self.slots = sorted(action_spec["slots"])
        self.num_arms = len(self.arms)
        self._policy_state_spec = policy_state_spec
        self.model_name = self.policy_state_spec.get(
            "model", "logistic_regression"
        )
        self.epsilon = self.policy_state_spec.get("epsilon", 1)
        self.epsilon_decay = self.policy_state_spec.get(
            "epsilon_decay", 0.9  # noqa: WPS432
        )
        self.epsilon_min = self.policy_state_spec.get(
            "epsilon_min", 0.01  # noqa: WPS432
        )  # noqa: WPS432

        self.num_features = self._policy_state_spec.get("num_feature", "all")

        if self.model_name == "logistic_regression":
            self._model_params = {
                "random_state": 0,
                "scoring": "roc_auc",
                "solver": "lbfgs",
                "max_iter": 1000,
                "cv": 5,
                "n_jobs": -1,
            }
            self._model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("drop_const", VarianceThreshold()),
                    ("selector", SelectKBest(k=self.num_features)),
                    ("scaler", StandardScaler()),
                    ("estimator", LogisticRegressionCV(**self._model_params)),
                ]
            )
        elif self.model_name == "xgboost":
            self._model = Pipeline(
                [
                 #   ("imputer", SimpleImputer(strategy="median")),
                 #   ("drop_const", VarianceThreshold()),
                 #   ("selector", SelectKBest(k=self.num_features)),
                    (
                        "estimator",
                        XGBClassifier(
                            use_label_encoder=False,
                            eval_metric="logloss",
                            random_state=0,
                            n_jobs=12
                        ),
                    ),
                ]
            )
        else:
            raise NotImplementedError

        self._dummy_min_sample = self._policy_state_spec.get(
            "dummy_min_sample", 20  # noqa: WPS432
        )
        self._dummy_model = Pipeline(
            [("estimator", DummyModel(self._dummy_min_sample))]
        )
        # if not specified, use noncontextual model
        if policy_state_spec.get("personalized", False):
            self._model = self._dummy_model
        self._fallback_model = FallbackModel(self._model, self._dummy_model)
        self._content_model = ModelSwitcher(self._fallback_model, "arm")
        self.optimize_slot = self._policy_state_spec.get(
            "optimize_slot", False
        )
        self.so = SlotOptimizer()
        self.position_bias = None
        self.cum_mean_rewards = []
        self.pre_trained_position_bias = self.so.pretrained_position_bias(
            self.num_arms
        )

    def action(self, contexts: np.array=None, policy_state: Dict={}) -> pd.DataFrame:
        """
        Sample a mapping from arms to slots.

        Args:
            A dataframe which indicates contexts.
                Columns are a list of contextual attributes.
                Index is `context_id` as int.
        Returns:
            A dataframe which indicates rankings.
                Index corresponds `context_id`.
                Columns indicates rank as int.
                Value is an arm for that rank.
        """
        contexts = pd.DataFrame(contexts)
        # predict
        expected = pd.DataFrame(index=contexts.index)
        # if not fitted, return random rankings
        if not hasattr(self._content_model, "_estimators"):
            r_rankings = [
                np.random.permutation(self.arms) for _ in range(len(expected))
            ]
            rankings = pd.DataFrame(r_rankings)
            # will need for OPE
            for arm in self.arms:
                contexts["arm"] = arm
                probs = np.random.rand()
                expected[arm] = probs
            self.expected = expected
        else:
            #contexts = contexts[self.features].copy()
            contexts = contexts.copy()
            for arm in self.arms:
                contexts["arm"] = arm
                # index class is 1 because it's binary classification
                probs = self._content_model.predict_proba(contexts)[:, 1]
                expected[arm] = probs
            self.expected = expected # OPE will use this
            sorted_arms = np.argsort(-expected.values, axis=1)
            rankings = pd.DataFrame(expected.columns.values[sorted_arms])
            if self.epsilon > 0:
                r_size = int(
                    np.max([1, (np.floor(len(contexts) * self.epsilon))])
                )
                # For single row inference case
                if r_size == 1:
                    r_size = 0 if self.epsilon < np.random.rand() else 1

                # pick index to randomize
                idx = np.random.choice(rankings.index, r_size)
                if len(idx) > 0:
                    rankings.iloc[idx] = [
                        np.random.permutation(self.arms) for _ in range(r_size)
                    ]
        rankings.columns = self.slots
        return rankings

    def update(self, action, reward, context=None, position=None) -> None:
        """
        Update from args of OPE
        """
        # currently we don't have sampling weights
        w = np.ones(len(action))
        features = pd.DataFrame(context)
        features['arm'] = action
        features = features.dropna(how='all', axis=1)
        features = features.fillna(0)
        # run update multiple times
        if self.optimize_slot:
            # call slot_optimizer
            position = pd.Series(position, name="slot_id")
            self.position_bias, self._content_model, self.content_bias = \
            self.so.EM_algorithm(
                features,
                reward,
                w,
                position,
                pd.DataFrame(action, columns=['arm']),
                self._content_model,
                position.unique(),
                slot_bias=self.position_bias,
                max_loop=20,  # noqa: WPS432
                rtol=1e-2,  # noqa: WPS432
                init_slot_prob=0.5,  # noqa: WPS432
                init_content_prob=0.5,  # noqa: WPS432
            )
        else:
            y = reward
            #w /= pd.Series(position).map(self.pre_trained_position_bias)
            #w *= pd.Series(position).map(self.pre_trained_position_bias)
            feature = pd.DataFrame(context)
            feature['arm'] = action
            feature = feature.dropna(how='all', axis=1)
            feature = feature.fillna(0)
            self._content_model.fit(feature, y, estimator__sample_weight=w)
        self.cum_mean_rewards.append(reward.sum() / len(action))
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        print("epsilon: {}", self.epsilon)


    def batch_update(self, data: pd.DataFrame) -> None:
        """
        Batch update for logistic epsilon greedy.
        
        Arg:
            data: a pandas DataFrame which has a following columns
                weight: float
                    sampling weights
                y: bool
                    target to optimize. e.g. click
                arm: str
                    name of arms
                position: int or str
                    id of positions
        """
        data = data.copy(deep=True)
        # run update multiple times
        if self.optimize_slot:
            # call slot_optimizer
            (
                features,
                y,
                w,
                slot_ids,
                content_ids,
                all_slot_ids,
            ) = self.so.process_df(data)

            self.features = features.columns.to_list()
            self.features.remove("arm")

            self.position_bias, self._content_model = self.so.EM_algorithm(
                features,
                y,
                w,
                slot_ids,
                content_ids,
                self._content_model,
                all_slot_ids,
                slot_bias=self.position_bias,
                max_loop=20,  # noqa: WPS432
                rtol=1e-2,  # noqa: WPS432
                init_slot_prob=0.5,  # noqa: WPS432
                init_content_prob=0.5,  # noqa: WPS432
            )
        else:
            w = data["weight"]
            y = data["y"]
            w /= data["position"].map(self.pre_trained_position_bias)
            features = data.drop(columns=["weight", "y", "position"])
            self.features = features.columns.to_list()
            self.features.remove("arm")
            self._content_model.fit(features, y, estimator__sample_weight=w)
        self.cum_mean_rewards.append(y.sum() / len(data))
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        print("epsilon: {}", self.epsilon)

    @property
    def time_step_spec(self):
        return self._time_step_spec

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def policy_state_spec(self):
        return self._policy_state_spec


