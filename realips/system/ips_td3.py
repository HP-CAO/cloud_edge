from realips.agent.td3 import TD3Agent, TD3AgentParams
from realips.trainer.trainer_td3 import TD3Trainer, TD3TrainerParams
from realips.system.ips import IpsSystem, IpsSystemParams


class IpsTD3Params(IpsSystemParams):
    def __init__(self):
        super().__init__()
        self.agent_params = TD3AgentParams()
        self.trainer_params = TD3TrainerParams()


class IpsTD3(IpsSystem):
    def __init__(self, params: IpsTD3Params):
        super().__init__(params)
        self.params = params
        if self.params.agent_params.add_actions_observations:
            self.shape_observations += self.params.agent_params.action_observations_dim
        self.agent = TD3Agent(params.agent_params, self.shape_observations, self.shape_targets, action_shape=1)
        self.agent.initial_model()
        self.trainer = TD3Trainer(params.trainer_params, self.agent)
        if self.params.stats_params.weights_path is not None:
            self.agent.load_weights(self.params.stats_params.weights_path)

    def test(self):
        self.evaluation_episode(self.agent)
