from realips.agent.ddpg import DDPGAgent, DDPGAgentParams
from realips.trainer.trainer_ddpg import DDPGTrainer, DDPGTrainerParams
from realips.system.ips import IpsSystem, IpsSystemParams
from realips.utils import states2observations
import numpy as np
import copy


class IpsDDPGParams(IpsSystemParams):
    def __init__(self):
        super().__init__()
        self.agent_params = DDPGAgentParams()
        self.trainer_params = DDPGTrainerParams()


class IpsDDPG(IpsSystem):
    def __init__(self, params: IpsDDPGParams):
        super().__init__(params)
        self.params = params
        if self.params.agent_params.add_actions_observations:
            self.shape_observations += self.params.agent_params.action_observations_dim
        self.agent = DDPGAgent(params.agent_params, self.shape_observations, self.shape_targets, shape_action=1)
        self.trainer = DDPGTrainer(params.trainer_params, self.agent)
        if self.params.stats_params.weights_path is not None:
            self.agent.load_weights(self.params.stats_params.weights_path)

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

                r = self.reward_fcn.reward(self.model_stats.observations, self.model_stats.targets, action, failed)

                self.trainer.store_experience(observations, self.model_stats.targets, action, r,
                                              observations_next, failed)

                self.model_stats.observations = copy.deepcopy(stats_observations_next)

                self.model_stats.measure(self.model_stats.observations, self.model_stats.targets, failed)
                self.model_stats.reward.append(r)

                self.trainer.optimize()

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

    def test(self):
        self.evaluation_episode(self.agent)
