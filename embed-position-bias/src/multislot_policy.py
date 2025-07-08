import abc


class MultislotPolicy(object):
    def __init__(self, time_step_spec, action_spec, policy_state_spec={}):
        """
        Base policy for multi-slot optimization.

        Args:
            action_spec: a dict which stores config of arms
            policy_state_spec: a dict which stores config of policy

        Attributes:
            arms: list of arms as strings
            num_arms: num of arms as integer
            _success: list of success counts as int
            _t: a value which discribes total steps so far
        """
        self.num_arms = None
        self.arms = None
        self._success = None
        self._t = None
        self._time_step_spec = time_step_spec
        self._action_spec = action_spec
        self._policy_state_spec = policy_state_spec

    def action(self, time_step, policy_state={}):
        """
        Sample a mapping from arms to slots.


        Returns:
            A dict {"arm": "slot"}. Order in the
            dict follows the ranking of arms and
            slots.
        """
        # return action(s) to pick
        pass

    def update(self, arm, arm_context, reward, user_context):
        """
        Update from args of contextual policy.
        """
        self._success[arm] += reward
        self._failure[arm] += 1 - reward
        self.t += 1
        self._time_step_spec["time_step"] = self.t
        sum_s = sum(self._success.values())
        sum_f = sum(self._failure.values())
        self.cum_mean_rewards.append(sum_s / (sum_s + sum_f))

    def batch_update():
        """
        Batch update of policy.
        """

        # train()

    @property
    def time_step_spec(self):
        return self._time_step_spec

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def policy_state_spec(self):
        return self._policy_state_spec
