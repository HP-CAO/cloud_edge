import numpy as np
import copy
from realips.utils import states2observations
from realips.monitor.mointor import ModelStatsParams, ModelStats
from realips.env.gym_physics import GymPhysics, GymPhysicsParams
from realips.env.reward import RewardParams, RewardFcn


class IpsSystemParams:
    def __init__(self):
        self.physics_params = GymPhysicsParams()
        self.reward_params = RewardParams()
        self.stats_params = ModelStatsParams()


class IpsSystem:
    def __init__(self, params: IpsSystemParams):
        self.params = params

        self.physics = GymPhysics(self.params.physics_params)
        self.model_stats = ModelStats(self.params.stats_params, self.physics)
        self.reward_fcn = RewardFcn(self.params.reward_params)
        self.shape_targets = self.model_stats.get_shape_targets()
        self.shape_observations = self.physics.get_shape_observations()

    def evaluation_episode(self, agent, ep=1):

        self.model_stats.init_episode()

        for step in range(self.params.stats_params.evaluation_steps):

            if self.params.stats_params.visualize_eval:
                self.physics.render()

            if agent.add_actions_observations:
                action_observations = np.zeros(shape=agent.action_observations_dim)
            else:
                action_observations = []

            observations = np.hstack((self.model_stats.observations, action_observations)).tolist()

            action = agent.get_exploitation_action(observations, self.model_stats.targets)

            states_next = self.physics.step(action)
            stats_observations_next, failed = states2observations(states_next)

            r = self.reward_fcn.reward(self.model_stats.observations, self.model_stats.targets, action, failed,
                                       pole_length=self.params.physics_params.length)

            self.model_stats.observations = copy.deepcopy(stats_observations_next)

            self.model_stats.measure(self.model_stats.observations, self.model_stats.targets,
                                     failed, pole_length=self.params.physics_params.length,
                                     distance_score_factor=self.params.reward_params.distance_score_factor)

            self.model_stats.reward.append(r)

            self.model_stats.cart_positions.append(self.physics.states[0])
            self.model_stats.pendulum_angele.append(self.physics.states[2])
            self.model_stats.actions.append(action)

            if failed:
                break

        distance_score_and_survived = float(
            self.model_stats.survived) * self.model_stats.get_average_distance_score()

        self.model_stats.evaluation_monitor_image(ep)

        self.physics.close()

        self.model_stats.evaluation_monitor_scalar(ep)

        return distance_score_and_survived
