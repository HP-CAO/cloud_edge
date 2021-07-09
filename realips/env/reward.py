class RewardParams:
    def __init__(self):
        self.distance_score_reward = 0.5
        self.action_penalty = 0.05
        self.crash_penalty = 10


class RewardFcn:

    def __init__(self, params: RewardParams, physics):
        self.params = params
        self.physics = physics

    def reward(self, observations, targets, action, terminal):
        """
        calculate reward
        :param observations: [pos, vel, sin_angle, cos_angle, angle_rate]
        :param targets: [pos_target, angle_target]
        :param action: action based on current states
        :param terminal: crash or not
        :return: a scalar value
        """

        distance_score = self.physics.get_distance_score(observations, targets)

        r = self.params.distance_score_reward * distance_score
        r -= self.params.action_penalty * action
        r -= self.params.crash_penalty * terminal

        return r
