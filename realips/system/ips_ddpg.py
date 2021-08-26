from realips.agent.ddpg import DDPGAgent, DDPGAgentParams
from realips.trainer.trainer_ddpg import DDPGTrainer, DDPGTrainerParams
from realips.system.ips import IpsSystem, IpsSystemParams
from realips.utils import states2observations
import numpy as np
import copy
import time


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
        self.agent.initial_model()
        if self.params.stats_params.weights_path is not None:
            self.agent.load_weights(self.params.stats_params.weights_path)

    def test(self):
        self.evaluation_episode(self.agent)