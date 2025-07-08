import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.exceptions import NotFittedError
from sklearn.metrics import log_loss


class SlotOptimizer:
    """
    SlotOptimizer will return slot biases (position biases).
    This class supports pretrained slot biases and
    Regression_based EM alogrithm.
    """

    def __init__(self):
        pass

    def pretrained_position_bias(self, num_position):
        # pre-estimated position biases by previous study in ebay
        # Reference : https://deepai.org/publication/direct-estimation-of-position-bias-for-unbiased-learning-to-rank-without-intervention
        position_biases = {0: 1}
        for i in range(2, num_position + 1):
            position_biases[i - 1] = min(1, 1 / np.log(i))
        # add later to avoid dividing by 0
        return position_biases

    def process_df(self, data):
        w = data["weight"]
        y = data["y"]
        content_ids = pd.Series(data["arm"])
        slot_ids = pd.Series(data["position"], name="slot_id")
        all_slot_ids = data["position"].unique()
        data.drop(columns=["weight", "y", "position"], inplace=True)
        X = data
        return X, y, w, slot_ids, content_ids, all_slot_ids

    def EM_algorithm(
        self,
        X: pd.DataFrame,
        y: np.array,
        w: np.array,
        slot_ids: pd.Series,
        content_ids: pd.Series,
        content_model,
        all_slot_ids: pd.Series,
        slot_bias=None,
        max_loop: int=20,
        rtol: float=1e-2,
        init_slot_prob: float=0.5,
        init_content_prob: float=0.5,
    ):
        """We iteratively update slot_bias as P(obs|slot) and
        content_mode as P(click|content,obs) by EM_algorithm
        Args:
            X: A pandas dataframe as data
            y: A numpy array as target labels
            w: A numpy array as sample_weight
            slot_ids: A pandas series as slot_ids
            content_ids: A pandas series as content_ids
            slot_bias: A series which inidicates initial slot_biases
            content_model: A scikit_learn estimator
            max_loop: A maximum number of iterations before parameters
            will be convergenced.
            rtol: A reletive tolerance as threshold of a termination
            condition.
        Reference:
            Position Bias Estimation for Unbiased Learning to
            Rank in Personal Search (2018)
            (doi:10.1145/3159652.3159732)
        """
        # get initial slot_bias and content_bias
        if slot_bias is not None:
            slot_bias_imp = slot_ids.map(slot_bias).values
        else:
            slot_bias_imp = init_slot_prob
            slot_bias = pd.Series(init_slot_prob, slot_ids.unique())
        try:
            content_bias_imp = content_model.predict_proba(X)[:, 1]
        except NotFittedError:
            content_bias_imp = init_content_prob
        for cnt_loop in range(max_loop):
            #print(f"EM algorithm: num of iter: {cnt_loop}")
            # equation 3 of 4.3.1 (3)
            # p of (relevance=1 and observe=0) given click=0, context
            # slot, content
            # Estep: calculate probabilities
            p_obs = y + (1 - y) * slot_bias_imp * (1 - content_bias_imp) / (
                1 - slot_bias_imp * content_bias_imp
            )
            p_unobs_rel = (
                (1 - slot_bias_imp)
                * content_bias_imp
                / (1 - slot_bias_imp * content_bias_imp)
            )
            # y_rel is a target for relevance while y is for click
            y_rel = y + (1 - y) * (np.random.rand(len(y)) < p_unobs_rel)
            # Mstep: train content_model, update parameters
            content_model.fit(X, y_rel, estimator__sample_weight=w)
            # update content_bias
            content_bias_imp = content_model.predict_proba(X)[:, 1]
            # update position bias by 4.3.1 (2) also assure index
            new_slot_bias = (
                pd.DataFrame({"p_obs": p_obs, "w": w}).set_index(slot_ids)
                .groupby("slot_id")
                .apply(lambda df: (df.w * df.p_obs).sum() / df.w.sum())
                .reindex(all_slot_ids)
                .fillna(0)
            )
            # there is an assumption about maximum position bias is always 1
            # new_slot_bias /= new_slot_bias.max()
            slot_bias = slot_bias.reindex(all_slot_ids).fillna(0)
            if np.allclose(
                new_slot_bias / new_slot_bias.sum(),
                slot_bias / slot_bias.sum(),
                rtol=rtol,
            ):
                break
            slot_bias = new_slot_bias 
            slot_bias_imp = slot_ids.map(new_slot_bias).values
        # log metrics
        #print("log_loss(y, slot_bias_imp * content_bias_imp): {}",
        #             log_loss(y, slot_bias_imp * content_bias_imp,
        #                      sample_weight=w, labels=[0, 1]))
        #print("log_loss(y_rel, content_bias_imp): {}",
        #             log_loss(y_rel, content_bias_imp, sample_weight=w,
        #                      labels=[0, 1]))
        content_bias = pd.DataFrame({'content_bias': content_bias_imp, 'arm': content_ids['arm']}).groupby('arm').mean()
        #print("mean of content_bias for contexts: {}", content_bias)
        #print("slot_biases: {}", slot_bias)
        return new_slot_bias, content_model, content_bias
