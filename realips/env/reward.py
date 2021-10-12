import numpy as np
import math


class RewardParams:
    def __init__(self):
        self.distance_score_reward = 0.5
        self.action_penalty = 0.05
        self.crash_penalty = 10
        self.distance_score_factor = 5


class RewardFcn:

    def __init__(self, params: RewardParams):
        self.params = params
        self.reward = self.distance_reward

    def distance_reward(self, observations, targets, action, terminal, pole_length):
        """
        calculate reward
        :param pole_length: the length of the pole
        :param observations: [pos, vel, sin_angle, cos_angle, angle_rate]
        :param targets: [pos_target, angle_target]
        :param action: action based on current states
        :param terminal: crash or not
        :return: a scalar value
        """

        distance_score = self.get_distance_score(observations, targets, pole_length, self.params.distance_score_factor)

        r = self.params.distance_score_reward * distance_score
        r -= self.params.action_penalty * action
        r -= self.params.crash_penalty * terminal

        return r

    @staticmethod
    def get_distance_score(observation, target, pole_length, distance_score_factor):
        """
        calculate reward
        :param pole_length: the length of the pole
        :param distance_score_factor: co-efficient of the distance score
        :param observation: [pos, vel, sin_angle, cos_angle, angle_rate]
        :param target: [pos_target, angle_target]
        """

        cart_position = observation[0]
        pendulum_angle_sin = observation[2]
        pendulum_angle_cos = observation[3]

        target_cart_position = target[0]
        target_pendulum_angle = target[1]

        pendulum_length = pole_length

        pendulum_tip_position = np.array(
            [cart_position + pendulum_length * pendulum_angle_sin, pendulum_length * pendulum_angle_cos])

        target_tip_position = np.array(
            [target_cart_position + pendulum_length * np.sin(target_pendulum_angle),
             pendulum_length * np.cos(target_pendulum_angle)])

        distance = np.linalg.norm(target_tip_position - pendulum_tip_position)

        return np.exp(-distance * distance_score_factor)  # distance [0, inf) -> score [1, 0)




