
from copy import deepcopy
from typing import Callable
from typing import Union

import pandas as pd
import numpy as np
import random
from tqdm import tqdm


def run_timeseries_simulation(
    bandit_feedback, policy, interval='hourly'
) -> np.ndarray:
    """Run an online bandit algorithm on the given logged bandit feedback data.
    We run batch job with houly or daily.

    Parameters
    ----------
    bandit_feedback: BanditFeedback
        Logged bandit data used in offline bandit simulation.

    policy: BanditPolicy
        Online bandit policy to be evaluated in offline bandit simulation (i.e., evaluation policy).

    interval: str
        Parameter for interval of batch. 'houly' or 'daily' wil. be allowed.

    Returns
    --------
    action_dist: array-like, shape (n_rounds, n_actions, len_list)
        Action choice probabilities (can be deterministic).
    """
    action_dist = np.empty([0, policy.num_arms, len(policy.slots)])
    if interval == 'hourly':
        idxts = pd.to_datetime(bandit_feedback['timestamp']).dt.strftime('%m/%d %H').value_counts(sort=False).sort_index().cumsum()
    elif isinstance(interval, int):
        r = len(bandit_feedback['action'])/interval
        idxts = pd.Series(np.arange(1, r+1) * interval).astype(int)
        if idxts.values[-1] > len(bandit_feedback['action']):
            idxts = idxts[:-1] 
    else:
        idxts = pd.to_datetime(bandit_feedback['timestamp']).dt.strftime('%m/%d').value_counts(sort=False).sort_index().cumsum()
    l = 0
    for i, r in enumerate(tqdm(idxts)):
        # pick actions
        selected_actions = policy.action(contexts=bandit_feedback['context'][l:r])
        # obtain rearranged actions 
        action_dist_sub = np.zeros((r-l, policy.num_arms, len(policy.slots)))
        for user in range(r-l):
            for position in range(len(policy.slots)):
                arm = selected_actions.loc[user][position]
                action_dist_sub[user][arm][position] = 1
        action_dist = np.vstack((action_dist, action_dist_sub))
        # update
        # sample_n = 10000
        policy.update(
            bandit_feedback["action"][:r],
            bandit_feedback["reward"][:r],
            context = bandit_feedback["context"][:r],
            position = bandit_feedback["position"][:r],
        )
        l = r
    return action_dist
