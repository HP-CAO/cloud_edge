import numpy as np
import copy
from realips.utils import states2observations
from realips.monitor.monitor import ModelStatsParams, ModelStats
from realips.env.gym_physics import GymPhysics, GymPhysicsParams
from realips.env.reward import RewardParams, RewardFcn


class IpsSystemParams:
    def __init__(self):
        self.physics_params = GymPhysicsParams()
        self.reward_params = RewardParams()
        self.stats_params = ModelStatsParams()
        self.agent_params = None


class IpsSystem:
    def __init__(self, params: IpsSystemParams):
        self.params = params

        self.physics = GymPhysics(self.params.physics_params)
        self.model_stats = ModelStats(self.params.stats_params, self.physics)
        self.reward_fcn = RewardFcn(self.params.reward_params)
        self.shape_targets = self.model_stats.get_shape_targets()
        self.shape_observations = self.physics.get_shape_observations()
        self.trainer = None
        self.agent = None

    def evaluation_episode(self, agent, ep=1):

        self.model_stats.init_episode()

        if agent.add_actions_observations:
            action_observations = np.zeros(shape=agent.action_observations_dim)
        else:
            action_observations = []

        for step in range(self.params.stats_params.max_episode_steps):

            if self.params.stats_params.visualize_eval:
                self.physics.render()

            observations = np.hstack((self.model_stats.observations, action_observations))

            action = agent.get_exploitation_action(observations, self.model_stats.targets)

            if self.params.agent_params.add_actions_observations:
                action_observations = np.append(action_observations, action)[1:]

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

    def train(self):

        ep = 0
        best_dsas = 0.0  # Best distance score and survived
        moving_average_dsas = 0.0

        while self.model_stats.total_steps < self.model_stats.params.total_steps:

            self.model_stats.init_episode()
            ep += 1
            step = 0

            if self.params.agent_params.add_actions_observations:
                action_observations = np.zeros(shape=self.params.agent_params.action_observations_dim)
            else:
                action_observations = []

            for step in range(self.params.stats_params.max_episode_steps):

                observations = np.hstack((self.model_stats.observations, action_observations)).tolist()

                action = self.agent.get_exploration_action(observations, self.model_stats.targets)

                if self.params.agent_params.add_actions_observations:
                    action_observations = np.append(action_observations, action)[1:]

                states_next = self.physics.step(action)

                stats_observations_next, failed = states2observations(states_next)

                observations_next = np.hstack((stats_observations_next, action_observations)).tolist()

                r = self.reward_fcn.reward(self.model_stats.observations, self.model_stats.targets, action, failed,
                                           pole_length=self.params.physics_params.length)

                self.trainer.store_experience(observations, self.model_stats.targets, action, r,
                                              observations_next, failed)

                self.model_stats.observations = copy.deepcopy(stats_observations_next)

                self.model_stats.measure(self.model_stats.observations, self.model_stats.targets, failed,
                                         pole_length=self.params.physics_params.length,
                                         distance_score_factor=self.params.reward_params.distance_score_factor)

                self.model_stats.reward.append(r)

                self.trainer.optimize()

                if self.model_stats.consecutive_on_target_steps > self.params.stats_params.on_target_reset_steps:
                    break

                if failed:
                    break

            self.model_stats.add_steps(step)
            self.model_stats.training_monitor(ep)
            self.agent.noise_factor_decay(self.model_stats.total_steps)

            if ep % self.params.stats_params.eval_period == 0:
                dsal = self.evaluation_episode(self.agent, ep)
                # self.agent.save_weights(self.params.stats_params.model_name + '_' + str(ep))
                moving_average_dsas = 0.95 * moving_average_dsas + 0.05 * dsal
                if moving_average_dsas > best_dsas:
                    self.agent.save_weights(self.params.stats_params.model_name + '_best')
                    best_dsas = moving_average_dsas

        self.agent.save_weights(self.params.stats_params.model_name)
