"""Base Interfaces for Bandit Algorithms."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional
from obp.policy import Random

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state, check_scalar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from obp.ope.estimators import InverseProbabilityWeighting, DirectMethod, DoublyRobust

from obp.policy.policy_type import PolicyType
from scipy.special import softmax
from scipy.optimize import brentq
from sklearn.linear_model import Lasso, LogisticRegression
from obp.ope import RegressionModel


@dataclass
class ModelSelectionPolicy(metaclass=ABCMeta):
    """Class of model selection bandit.
    For each batch, this policy calculate evaluation of candidate policies 
    and then pick up the best policy.
    It corresponds to MetaGreedyPolicy in the paper.

    Parameters
    ----------
    dim: int
        Number of dimensions of context vectors.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking interface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.

    random_state: int, default=None
        Controls the random seed in sampling actions.
    """

    dim: int
    n_actions: int
    policies: list
    mode: str = 'OPE'
    len_list: int = 1
    batch_size: int = 1
    random_state: Optional[int] = None
    epsilon: float=0.05

    def __post_init__(self) -> None:
        """Initialize class.
        
        This method is called after the dataclass __init__ method. It performs
        additional initialization tasks such as checking scalar values and setting
        up initial variables.

        Variables:
        ----------
        policies: list
            A list of candidate policy instances.
        """
        check_scalar(self.dim, "dim", int, min_val=1)
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1, max_val=self.n_actions)
        check_scalar(self.batch_size, "batch_size", int, min_val=1)
        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.reward_lists = [[] for _ in np.arange(self.n_actions)]
        self.context_lists = [[] for _ in np.arange(self.n_actions)]
        self.policy = Random(self.n_actions)
        self.selected_policies = []

    def initialize(self) -> None:
        """Initialize policy parameters.
        
        This method resets the policy parameters, including the trial count,
        random state, action counts, reward lists, and context lists.
        """
        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.reward_lists = [[] for _ in np.arange(self.n_actions)]
        self.context_lists = [[] for _ in np.arange(self.n_actions)]

    def select_action(self, contexts: np.ndarray) -> np.ndarray:
        """Select a list of actions.
        
        This method selects which policy to pick and then samples actions based on
        the selected policy.

        Parameters:
        ----------
        contexts: np.ndarray
            Context vectors for the current decision.

        Returns:
        -------
        np.ndarray
            Selected actions.
        """
        if repr(self.policy.policy_type) == 'PolicyType.CONTEXTUAL':
            return self.policy.select_action(contexts)[0]
        else:
            return self.policy.select_action()[0]

    def update_params(self, action: float, reward: float, context: np.ndarray) -> None:
        """Update policy parameters.
        
        This method updates the parameters of the selected policy based on the
        provided action, reward, and context.

        Parameters:
        ----------
        action: float
            The action taken.

        reward: float
            The reward received.

        context: np.ndarray
            The context vector.
        """
        self.policy = self.select_policy(action, reward, context)
        for i in range(len(self.policies)):
            for saction, sreward, scontext in zip(action, reward, context):
                if repr(self.policies[i].policy_type) == 'PolicyType.CONTEXTUAL':
                    self.policies[i].update_params(
                        saction, sreward, scontext[np.newaxis, :])
                else:
                    self.policies[i].update_params(
                        saction, sreward)

    def select_policy(self, action, reward, context):
        """Select the best policy based on the evaluation mode.
        
        This method evaluates the candidate policies and selects the one with the
        highest evaluation score.

        Parameters:
        ----------
        action: float
            The action taken.

        reward: float
            The reward received.

        context: np.ndarray
            The context vector.

        Returns:
        -------
        Policy
            The selected policy.
        """
        if self.mode == 'OPE':
            evaluations = self.calculateOPE(action, reward, context)
        elif self.mode == 'feature_surface':
            evaluations = self.check_feature_surface(action, reward, context)
        elif self.mode == 'accuracy':
            evaluations = self.check_accuracy(action, reward, context)
        elif self.mode == 'logloss':
            evaluations = [-1*x for x in self.check_logloss(action, reward, context)]
        else:
            ValueError("Invalid name of config['mode']")
        rand = np.random.rand()
        if rand < self.epsilon:
            max_index = np.random.choice(len(evaluations))
        else:
            max_index = np.argmax(np.array(evaluations))
        self.selected_policies.append(max_index)
        return self.policies[max_index]

    def calculateOPE(self, action, reward, context):
        """Calculate expected reward for the most recent observation using OPE.
        
        This method calculates the expected reward for each candidate policy using
        Off-Policy Evaluation (OPE).

        Parameters:
        ----------
        action: float
            The action taken.

        reward: float
            The reward received.

        context: np.ndarray
            The context vector.

        Returns:
        -------
        list
            List of expected rewards for each candidate policy.
        """
        opes = []
        for i in range(len(self.policies)):
            actions = []
            for scontext in context:
                if repr(self.policies[i].policy_type) == 'PolicyType.CONTEXTUAL':
                    actions.append(
                        self.policies[i].select_action(scontext)[0])
                else:
                    actions.append(
                        self.policies[i].select_action()[0])
            action_dist = np.zeros((len(actions), self.n_actions))
            for j in range(len(actions)):
                action_dist[j, actions[j]] = 1
            action_dist = action_dist[:, :, np.newaxis]
            ipw = InverseProbabilityWeighting()
            ope = ipw.estimate_policy_value(
                reward=reward,
                action=action,
                action_dist=action_dist,
                pscore=self.calc_pscore(action)
            )
            opes.append(ope)
        return opes

    def check_feature_surface():
        """Placeholder for checking feature surface."""
        pass

    def check_accuracy(self, action_log, reward_log, context):
        """
        Calculate accuracy for the most recent observation using OPE.
        
        This method calculates the expected reward for each candidate policy using
        Off-Policy Evaluation (OPE).
        Only available only when policy has internal predictor.

        Parameters:
        ----------
        action: float
            The action taken.

        reward: float
            The reward received.

        context: np.ndarray
            The context vector.

        Returns:
        -------
        list
            List of expected rewards for each candidate policy.
        """

        accuracy_scores = []
        for i in range(len(self.policies)):
            actions = []
            y_pred = []
            for scontext in context:
                if repr(self.policies[i].policy_type) == 'PolicyType.CONTEXTUAL':
                    expected_vals = self.policies[i].expected_reward(scontext)
                else:
                    expected_vals = self.policies[i].expected_reward()
                action = np.argmax(expected_vals)
                actions.append(action)
                y_pred.append(expected_vals[action])
            y_pred = np.array(y_pred)
            matched_idx = np.array(actions) == action_log
            # check_accuracy, sample reward from expected_reward and calc it.
            rewards = np.random.rand(np.sum(matched_idx)) < y_pred[matched_idx]
            accuracy_scores.append(
                accuracy_score(reward_log[matched_idx], rewards))
            # check_logloss -> Not implemented yet
        return accuracy_scores


    def check_logloss(self, action_log, reward_log, context):
        """
        Calculate accuracy for the most recent observation using OPE.
        
        This method calculates the expected reward for each candidate policy using
        Off-Policy Evaluation (OPE).
        Only available only when policy has internal predictor.

        Parameters:
        ----------
        action: float
            The action taken.

        reward: float
            The reward received.

        context: np.ndarray
            The context vector.

        Returns:
        -------
        list
            List of expected rewards for each candidate policy.
        """
        loglosses = []
        for i in range(len(self.policies)):
            actions = []
            y_pred = []
            for scontext in context:
                if repr(self.policies[i].policy_type) == 'PolicyType.CONTEXTUAL':
                    expected_vals = self.policies[i].expected_reward(scontext)
                else:
                    expected_vals = self.policies[i].expected_reward()
                action = np.argmax(expected_vals)
                actions.append(action)
                y_pred.append(expected_vals[action])
            y_pred = np.array(y_pred)
            matched_idx = np.array(actions) == action_log
            # check_accuracy, sample reward from expected_reward and calc it.
            try:
                loglosses.append(
                    log_loss(reward_log[matched_idx], y_pred[matched_idx]))
            except:
                print("can't calculate logloss, put random val instead.")
                loglosses.append(np.random.rand())
            # check_logloss -> Not implemented yet
        return loglosses


    def calc_pscore(self, action):
        """Calculate the propensity score from the data.
        
        This method calculates the propensity score for each action based on the
        observed action distribution.

        Parameters:
        ----------
        action: float
            The action taken.

        Returns:
        -------
        np.ndarray
            Propensity scores for each action.
        """
        pscore_dict = (pd.Series(action).value_counts() / len(action)).to_dict()
        return pd.Series(action).replace(pscore_dict).values


@dataclass
class MetaExp3P(metaclass=ABCMeta):
    """
    Implementation of https://arxiv.org/pdf/2003.01704
    """
    dim: int
    n_actions: int
    policies: list
    len_list: int = 1
    batch_size: int = 1
    random_state: Optional[int] = None
    p: float = 0.05 #exploration rate

    def __post_init__(self) -> None:
        check_scalar(self.dim, "dim", int, min_val=1)
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1, max_val=self.n_actions)
        check_scalar(self.batch_size, "batch_size", int, min_val=1)
        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.reward_lists = [[] for _ in np.arange(self.n_actions)]
        self.context_lists = [[] for _ in np.arange(self.n_actions)]
        self.policy = Random(self.n_actions)
        self.selected_policies = []
        self.M = len(self.policies)
        self.p_j_t = np.ones(self.M) / self.M  # Initialize probabilities of choosing policy
        self.Re_j_t = np.zeros(self.M)  # Initialize estimated cumulative gains
        self.batch_its = []

    def initialize(self) -> None:
        pass
    
    def select_action(self, contexts):
        # Sample an algorithm according to the current probabilities
        # rewrite original setting for batch environment
        it = np.random.choice(self.M, p=self.p_j_t)
        self.batch_its.append(it)
        policy = self.policies[it]
        self.selected_policies.append(it)
        if repr(policy.policy_type) == 'PolicyType.CONTEXTUAL':
            return policy.select_action(contexts)[0]
        else:
            return policy.select_action()[0]


    def update_params(self, action: float, reward: float, context: np.ndarray) -> None:
        # update params of candidate policies
        for saction, sreward, scontext, it in zip(action, reward, context, self.batch_its):
            if repr(self.policies[it].policy_type) == 'PolicyType.CONTEXTUAL':
                self.policies[it].update_params(
                    saction, sreward, scontext[np.newaxis, :])
            else:
                self.policies[it].update_params(
                    saction, sreward)
        
            # Receive feedback from the chosen base algorithm
            rt = sreward 
            # Compute the estimated gain for each base algorithm
            re_t = np.zeros(self.M)
            for j in range(self.M):
                re_t[j] = rt if j == it else 0
                re_t[j] = (re_t[j] + self.p/2) / self.p_j_t[j] if self.p_j_t[j] > 0 else 0
            self.Re_j_t += re_t
            # Update the estimated cumulative gain
            self.p_j_t = (1 - self.p) * softmax(self.Re_j_t) + self.p
            self.p_j_t /= np.sum(self.p_j_t)
            self.batch_its = []



@dataclass
class MetaCORRAL(metaclass=ABCMeta):
    """
    Implementation of https://arxiv.org/pdf/2003.01704
    """
    dim: int
    n_actions: int
    T: int
    policies: list
    len_list: int = 1
    batch_size: int = 1
    random_state: Optional[int] = None
    eta: float = 0.4 #exploration rate

    def __post_init__(self) -> None:
        check_scalar(self.dim, "dim", int, min_val=1)
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1, max_val=self.n_actions)
        check_scalar(self.batch_size, "batch_size", int, min_val=1)
        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.reward_lists = [[] for _ in np.arange(self.n_actions)]
        self.context_lists = [[] for _ in np.arange(self.n_actions)]
        self.policy = Random(self.n_actions)
        self.selected_policies = []
        self.M = len(self.policies)
        self.p_t = np.ones(self.M) / self.M  # Initialize probabilities of choosing policy
        self.ro = 2*self.M*np.ones(self.M)  # Initialize estimated cumulative gains
        self.p_t_ubar = 1/self.ro
        self.batch_its = []
        self.gamma = 1/self.T  # never know T in the practical setting
        self.beta = np.exp(1/np.log(self.T))
        self.eta = self.eta * np.ones(self.M)
        self.num_error_brentq = 0


    def initialize(self) -> None:
        pass
    
    def select_action(self, contexts):
        # Sample an algorithm according to the current probabilities
        # rewrite original setting for batch environment
        self.j_t = np.random.choice(self.M, p=self.p_t)
        self.batch_its.append(self.j_t)
        policy = self.policies[self.j_t]
        self.selected_policies.append(self.j_t)
        if repr(policy.policy_type) == 'PolicyType.CONTEXTUAL':
            return policy.select_action(contexts)[0]
        else:
            return policy.select_action()[0]


    def update_params(self, action: float, reward: float, context: np.ndarray) -> None:
        # update params of candidate policies
        for saction, sreward, scontext, it in zip(action, reward, context, self.batch_its):
            if repr(self.policies[it].policy_type) == 'PolicyType.CONTEXTUAL':
                self.policies[it].update_params(
                    saction, sreward, scontext[np.newaxis, :])
            else:
                self.policies[it].update_params(
                    saction, sreward)
            self.j_t = it
            self.p_t, self.eta, self.p_t_ubar, self.ro = self.corral_update(sreward)
        self.batch_its = []
    
    def log_barrier_omd(self, eta_t, p_t, l_t):
        """
        Log-Barrier-OMD algorithm implementation in Python.
        
        Parameters:
        eta_t (numpy array): Learning rate vector.
        p_t (numpy array): Previous distribution.
        l_t (numpy array): Current loss.
        
        Returns:
        numpy array: Updated distribution.
        """
        def equation_to_solve(lambda_val):
            return np.sum(1 / (1 / p_t + eta_t * (l_t - lambda_val))) - 1
    
        # Find lambda in the range [min(l_t), max(l_t)]
        lambda_min = np.min(l_t)
        lambda_max = np.max(l_t)
        try:
            lambda_opt = brentq(equation_to_solve, lambda_min, lambda_max)
        except ValueError:
            # aboart update
            self.num_error_brentq += 1
            return p_t
        # Update the distribution
        p_t_plus_1 = 1 / (1 / p_t + eta_t * (l_t - lambda_opt))
        p_t_plus_1 /= p_t_plus_1.sum()
        return p_t_plus_1
    
    def corral_update(self, r_t):
        """
        CORRAL-Update algorithm implementation in Python.
        
        Parameters:
        eta_t (numpy array): Learning rate vector.
        p_t (numpy array): Current distribution.
        p_lower (numpy array): Lower bound distribution.
        r_t (numpy array): Current loss.
        gamma (float): Mixing parameter.
        beta (float): Learning rate adjustment factor.
        
        Returns:
        tuple: Updated distribution (p_t_plus_1), updated learning rate vector (eta_t_plus_1), and updated loss range (rho_t_plus_1).
        """
        M = len(self.p_t)
        r_t *= -1 # original paper says rt is loss.
        # Step 3: Update pt+1 using Log-Barrier-OMD
        e_j_t = np.zeros(M)
        e_j_t[self.j_t] = 1
        p_t_plus_1 = self.log_barrier_omd(self.eta, self.p_t, r_t / self.p_t[self.j_t] * e_j_t)
        
        # Step 4: Set pt+1 = (1 − γ)pt+1 + γ / M
        p_t_plus_1 = (1 - self.gamma) * p_t_plus_1 + self.gamma / M
        # Initialize 
        eta_t = self.eta
        eta_t_plus_1 = np.copy(eta_t)
        p_t_plus_1_ubar = np.zeros(M)
        p_t = self.p_t
        rho_t_plus_1 = np.zeros(M)
        
        # Step 5-10: Update eta_t_plus_1 and rho_t_plus_1
        # consider lower p
        for j in range(M):
            if self.p_t_ubar[j] > p_t_plus_1[j]:
                p_t_plus_1_ubar[j] = p_t_plus_1[j]/2
                eta_t_plus_1[j] = self.beta * eta_t[j]
            else:
                p_t_plus_1_ubar[j] = self.p_t_ubar[j]
                eta_t_plus_1[j] = eta_t[j]
            
            rho_t_plus_1[j] = 1 / p_t_plus_1_ubar[j]
        return p_t_plus_1, eta_t_plus_1, p_t_plus_1_ubar, rho_t_plus_1



@dataclass
class MetaEXPOPE(metaclass=ABCMeta):
    """Class of model selection bandit.
    For each batch, this policy calculate evaluation of candidate policies 
    and then pick up the best policy.

    Parameters
    ----------
    dim: int
        Number of dimensions of context vectors.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking interface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.

    random_state: int, default=None
        Controls the random seed in sampling actions.
    """

    dim: int
    T: int
    n_actions: int
    policies: list
    ope: str = 'IPW'
    len_list: int = 1
    batch_size: int = 1
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize class.
        
        This method is called after the dataclass __init__ method. It performs
        additional initialization tasks such as checking scalar values and setting
        up initial variables.

        Variables:
        ----------
        policies: list
            A list of candidate policy instances.
        """
        check_scalar(self.dim, "dim", int, min_val=1)
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1, max_val=self.n_actions)
        check_scalar(self.batch_size, "batch_size", int, min_val=1)
        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.reward_lists = [[] for _ in np.arange(self.n_actions)]
        self.context_lists = [[] for _ in np.arange(self.n_actions)]
        self.policy = Random(self.n_actions)
        self.size_context = self.dim
        # assign uniform prob for the probability of selecting policy.
        self.q = np.ones(len(self.policies), dtype=int)/len(self.policies)
        self.selected_policies = []
        self.gamma = 0.05
        self.n_policies = len(self.policies)
        self.evaluations = np.ones(self.n_policies) * 1/self.n_policies
        self.initialize()
        self.all_action = np.empty(0, dtype='int')
        self.all_context = np.empty((0, self.size_context))
        self.all_reward = np.empty(0, dtype='int')

    def initialize(self) -> None:
        """Initialize policy parameters.
        
        This method resets the policy parameters, including the trial count,
        random state, action counts, reward lists, and context lists.
        """
        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.reward_lists = [[] for _ in np.arange(self.n_actions)]
        self.context_lists = [[] for _ in np.arange(self.n_actions)]
        self.regression_model = RegressionModel(
            n_actions=self.n_actions,
            action_context=np.identity(self.n_actions), # currently assume OHE for context
            base_model=LogisticRegression(random_state=42),
        )

    def select_action(self, contexts: np.ndarray) -> np.ndarray:
        """Select a list of actions.
        
        This method selects which policy to pick and then samples actions based on
        the selected policy.

        Parameters:
        ----------
        contexts: np.ndarray
            Context vectors for the current decision.

        Returns:
        -------
        np.ndarray
            Selected actions.
        """
        # bernoulli part
        alpha_t = np.random.binomial(1, self.gamma)
        if alpha_t:
            return np.random.choice(self.n_actions)
        else:
            policy_index = np.random.choice(len(self.policies), p=self.evaluations)
            self.selected_policies.append(policy_index)
            self.policy = self.policies[policy_index]
            if repr(self.policy.policy_type) == 'PolicyType.CONTEXTUAL':
                return self.policy.select_action(contexts)[0]
            else:
                return self.policy.select_action()[0]

    def update_params(self, action: float, reward: float, context: np.ndarray) -> None:
        """Update policy parameters.
        
        This method updates the parameters of the selected policy based on the
        provided action, reward, and context.

        Parameters:
        ----------
        action: float
            The action taken.

        reward: float
            The reward received.

        context: np.ndarray
            The context vector.
        """        
        self.select_policy(action, reward, context)
        # update_policy
        for i in range(len(self.policies)):
            for saction, sreward, scontext, vaction in zip(
                    action, reward, context, self.virtual_actions_all[i]):
                """
                if saction != vaction:
                    continue
                """
                if repr(self.policies[i].policy_type) == 'PolicyType.CONTEXTUAL':
                    self.policies[i].update_params(
                        saction, sreward, scontext[np.newaxis, :])
                else:
                    self.policies[i].update_params(
                        saction, sreward)   

    def select_policy(self, action, reward, context):
        """Select the best policy based on the evaluation mode.
        
        This method evaluates the candidate policies and selects the one with the
        highest evaluation score.

        Parameters:
        ----------
        action: float
            The action taken.

        reward: float
            The reward received.

        context: np.ndarray
            The context vector.

        Returns:
        -------
        Policy
            The selected policy.
        """
        
        evaluations = self.calculateOPE(action, reward, context)
        self.evaluations = softmax(evaluations)
        """
        policy_index = np.random.choice(len(self.policies), p=evaluations)
        self.selected_policies.append(policy_index)
        return policy_index
        """
        return None

    def calculateOPE(self, action, reward, context):
        """Calculate expected reward for the most recent observation using OPE.
        
        This method calculates the expected reward for each candidate policy using
        Off-Policy Evaluation (OPE).

        Parameters:
        ----------
        action: float
            The historical action which is actually taken.

        reward: float
            The reward received.

        context: np.ndarray
            The context vector.

        Returns:
        -------
        list
            List of expected rewards for each candidate policy.
        """
        
        opes = []
        self.virtual_actions_all = {}
        for i in range(len(self.policies)):
            virtual_actions = []
            for scontext in context:
                if repr(self.policies[i].policy_type) == 'PolicyType.CONTEXTUAL':
                    virtual_actions.append(
                        self.policies[i].select_action(scontext)[0])
                else:
                    virtual_actions.append(
                        self.policies[i].select_action()[0])
            action_dist = np.zeros((len(virtual_actions), self.n_actions))
            for j in range(len(virtual_actions)):
                action_dist[j, virtual_actions[j]] = 1
            action_dist = action_dist[:, :, np.newaxis]
            if self.ope == 'IPW':
                ipw = InverseProbabilityWeighting()
                ope = ipw.estimate_policy_value(
                    reward=reward,
                    action=action,
                    action_dist=action_dist,
                    pscore=self.calc_pscore(action)
                )
            elif self.ope == 'DM':
                estimated_rewards_by_reg_model = self.prepare_reg_model(action, reward, context)
                dm = DirectMethod()
                ope = dm.estimate_policy_value(
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                )
            elif self.ope == 'DR':
                estimated_rewards_by_reg_model = self.prepare_reg_model(action, reward, context)
                dr = DoublyRobust()
                ope = dr.estimate_policy_value(
                    reward=reward,
                    action=action,
                    action_dist=action_dist,
                    pscore=self.calc_pscore(action),
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                )
            opes.append(ope)
            self.virtual_actions_all[i] = virtual_actions            
        return opes

    def prepare_reg_model(self, action, reward, context):
        self.all_action = np.hstack((self.all_action, action))
        self.all_context = np.vstack((self.all_context, context))
        self.all_reward = np.hstack((self.all_reward, reward))
        self.regression_model.fit(
            action=self.all_action,
            context=self.all_context,
            reward=self.all_reward,
        )
        estimated_rewards_by_reg_model = self.regression_model.predict(context)
        return estimated_rewards_by_reg_model

    def check_feature_surface():
        """Placeholder for checking feature surface."""
        pass

    def calc_pscore(self, action):
        """Calculate the propensity score from the data.
        
        This method calculates the propensity score for each action based on the
        observed action distribution.

        Parameters:
        ----------
        action: float
            The action taken.

        Returns:
        -------
        np.ndarray
            Propensity scores for each action.
        """
        pscore_dict = (pd.Series(action).value_counts() / len(action)).to_dict()
        return pd.Series(action).replace(pscore_dict).values

