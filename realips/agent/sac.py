from realips.agent.td3 import TD3Agent, TD3AgentParams


class SacAgentParams(TD3AgentParams):
    def __init__(self):
        super(SacAgentParams, self).__init__()


class SacAgent(TD3Agent):
    def __init__(self, params: TD3AgentParams, observations_shape=5, target_shape=2, action_shape=1, on_edge=False):
        super(SacAgent, self).__init__(params, observations_shape, target_shape, action_shape, on_edge)
        self.critic_2 = self.build_critic("sac_2nd_critic")
        self.critic_target_2 = self.build_critic("sac_2nd_critic_target")
