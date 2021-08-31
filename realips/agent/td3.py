import os
from realips.agent.base import BaseAgent, BaseAgentParams


class TD3AgentParams(BaseAgentParams):
    def __init__(self):
        super(TD3AgentParams, self).__init__()


class TD3Agent(BaseAgent):
    def __init__(self, params: TD3AgentParams, observations_shape=5, target_shape=2, action_shape=1, on_edge=False):
        super(TD3Agent, self).__init__(params, observations_shape, target_shape, action_shape, on_edge)

    def initial_model(self):

        self.actor = self.build_actor("normal-")

        if not self.on_edge:
            self.actor_target = self.build_actor("target-")
            self.critic = self.build_critic("normal-")
            self.critic_target = self.build_critic("target-")
            self.critic_2 = self.build_critic("Td3_2nd_critic")
            self.critic_target_2 = self.build_critic("Td3_2nd_critic_target")
            self.hard_update()

    def soft_update(self):
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.actor_target.get_weights()
        self.actor_target.set_weights(
            [w_new * self.params.soft_alpha + w_old * (1. - self.params.soft_alpha) for w_new, w_old
             in zip(actor_weights, actor_target_weights)])

        critic_weights = self.critic.get_weights()
        critic_target_weights = self.critic_target.get_weights()
        self.critic_target.set_weights(
            [w_new * self.params.soft_alpha + w_old * (1. - self.params.soft_alpha) for w_new, w_old
             in zip(critic_weights, critic_target_weights)])

        critic_2_weights = self.critic_2.get_weights()
        critic_target_2_weights = self.critic_target_2.get_weights()
        self.critic_target_2.set_weights(
            [w_new * self.params.soft_alpha + w_old * (1. - self.params.soft_alpha) for w_new, w_old
             in zip(critic_2_weights, critic_target_2_weights)])

    def hard_update(self):
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic_target_2.set_weights(self.critic_2.get_weights())

    def save_weights(self, model_name):
        self.actor.save_weights('./models/' + model_name + '_Td3/actor_weights')
        self.critic.save_weights('./models/' + model_name + '_Td3/critic_weights')
        self.critic_2.save_weights('./models/' + model_name + '_Td3/critic_2_weights')
        self.actor_target.save_weights('./models/' + model_name + '_Td3/actor_target_weights')
        self.critic_target.save_weights('./models/' + model_name + '_Td3/critic_target_weights')
        self.critic_target_2.save_weights('./models/' + model_name + '_Td3/critic_target_2_weights')

    def load_weights(self, path_to_weights):
        print("loading pretrained weights......")

        if not os.path.exists(path_to_weights):
            raise IOError("Weights path not exist")

        path_to_actor = path_to_weights + 'actor_weights'
        path_to_critic = path_to_weights + 'critic_weights'
        path_to_critic_2 = path_to_weights + 'critic_2_weights'

        self.actor.load_weights(path_to_actor)
        self.critic.load_weights(path_to_critic)
        self.critic_2.load_weights(path_to_critic_2)
        self.hard_update()
