import math
import tensorflow as tf
import os
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from realips.utils import OrnsteinUhlenbeckActionNoise
import numpy as np


class DDPGAgentParams(object):

    def __init__(self):
        # critic config
        self.critic_dense1_obs = 256
        self.critic_dense2_obs = 128
        self.critic_dense1_act = 128
        self.critic_dense1 = 256
        self.critic_dense2 = 128

        # action config
        self.actor_dense1 = 256
        self.actor_dense2 = 128
        self.actor_dense3 = 64

        self.soft_alpha = 0.005
        self.action_noise_factor = 1
        self.action_noise_half_decay_time = 1e6

        self.add_actions_observations = True
        self.action_observations_dim = 5


class DDPGAgent:

    def __init__(self, params: DDPGAgentParams, shape_observations=5, shape_targets=2, shape_action=1, on_edge=False):
        self.params = params
        self.shape_observations = shape_observations
        self.shape_targets = shape_targets
        self.shape_action = shape_action
        self.on_edge = on_edge
        self.action_noise = OrnsteinUhlenbeckActionNoise(self.shape_action)
        self.action_noise_factor = params.action_noise_factor
        self.add_actions_observations = self.params.add_actions_observations
        self.action_observations_dim = self.params.action_observations_dim

    def initial_model(self):
        self.actor = self.build_actor("normal-")
        if not self.on_edge:
            self.critic = self.build_critic("normal-")
            self.critic_target = self.build_critic("target-")
            self.actor_target = self.build_actor("target-")
            self.hard_update()

    def build_critic(self, name):
        """
        return: Model of critic neural network
        """
        # observations input branch
        input_observations = Input(shape=(self.shape_observations,), name=name + 'observations_input', dtype=tf.float32)
        input_targets = Input(shape=(self.shape_targets,), name=name + 'targets_input', dtype=tf.float32)
        input_concat = tf.concat([input_observations, input_targets], 1, name=name + 'input_concat')

        dense1_obs = Dense(self.params.critic_dense1_obs, activation='relu', name=name + 'dense1_obs')(input_concat)
        dense2_obs = Dense(self.params.critic_dense2_obs, activation='relu', name=name + 'dense2_obs')(dense1_obs)

        # action input branch
        input_action = Input(shape=(self.shape_action,), name=name + 'action_input', dtype=tf.float32)
        dense1_act = Dense(self.params.critic_dense1_act, activation='relu', name=name + 'dense1_act')(input_action)

        # merge two branches
        mid_concat = tf.concat([dense2_obs, dense1_act], 1, name=name + 'mid_concat')
        dense1_critic = Dense(self.params.critic_dense1, activation='relu', name=name + 'dense1_critic')(mid_concat)
        dense2_critic = Dense(self.params.critic_dense2, activation='relu', name=name + 'dense2_critic')(dense1_critic)
        output_value = Dense(1, activation=None, name=name + 'output_critic')(dense2_critic)

        # generate critic network
        model_critic = Model(inputs=[input_observations, input_targets, input_action], outputs=output_value,
                             name=name + 'critic')

        return model_critic

    def build_actor(self, name):
        """
        return: model of actor neural network
        """
        input_observations = Input(shape=(self.shape_observations,), name=name + 'observations_input',
                                   dtype=tf.float32)
        input_targets = Input(shape=(self.shape_targets,), name=name + 'targets_input', dtype=tf.float32)
        input_concat = tf.concat([input_observations, input_targets], 1, name=name + 'input_concat')

        dense1_actor = Dense(self.params.actor_dense1, activation='relu', name=name + 'dense1_actor')(input_concat)
        dense2_actor = Dense(self.params.actor_dense2, activation='relu', name=name + 'dense2_actor')(dense1_actor)
        dense3_actor = Dense(self.params.actor_dense3, activation='relu', name=name + 'dense3_actor')(dense2_actor)
        output_action = Dense(self.shape_action, activation='tanh', name=name + 'action_output')(dense3_actor)

        # generate actor network
        model_actor = Model(inputs=[input_observations, input_targets], outputs=output_action, name=name + 'actor')

        return model_actor

    def get_exploration_action(self, observations, targets):
        """
        get the action from actor and add exploration noise
        :param targets: a list
        :param observations: a list
        :return: a scalar value
        """
        var = self.action_noise_factor
        observations = tf.expand_dims(observations, 0) # add batch dim
        targets = tf.expand_dims(targets, 0)
        action = self.actor([observations, targets]).numpy().squeeze()
        new_action = action + self.action_noise.sample() * var
        action_saturated = np.clip(new_action, -1, 1).squeeze()

        return action_saturated

    def get_exploitation_action(self, observations, targets):
        """
        get the action from actor for inference
        :param targets:
        :param observations:
        :return: action scalar
        """
        observations = tf.expand_dims(observations, 0)
        targets = tf.expand_dims(targets, 0)
        action = self.actor([observations, targets]).numpy().squeeze()
        return action

    def get_exploitation_action_target(self, observations, targets):
        """
        get the action from actor_target, used during training
        :param targets:
        :param observations:
        :return: action tensor (batch_size x 1)
        """
        action = self.actor_target([observations, targets])

        return action

    def hard_update(self):
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

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

    def get_actor_weights(self):
        return self.actor.get_weights()

    def set_actor_weights(self, weights):
        self.actor.set_weights(weights)

    def save_weights(self, model_name):
        self.actor.save_weights('./models/' + model_name + '_DDPG/actor_weights')
        self.critic.save_weights('./models/' + model_name + '_DDPG/critic_weights')
        self.actor_target.save_weights('./models/' + model_name + '_DDPG/actor_target_weights')
        self.critic_target.save_weights('./models/' + model_name + '_DDPG/critic_target_weights')

    def load_weights(self, path_to_weights):
        print("loading pretrained weights......")

        if not os.path.exists(path_to_weights):
            raise IOError("Weights path not exist")

        path_to_actor = path_to_weights + 'actor_weights'
        path_to_critic = path_to_weights + 'critic_weights'
        path_to_actor_target = path_to_weights + 'actor_target_weights'
        path_to_critc_target = path_to_weights + 'critic_target_weights'

        self.actor.load_weights(path_to_actor)
        self.critic.load_weights(path_to_critic)
        self.actor_target.load_weights(path_to_actor_target)
        self.critic_target.load_weights(path_to_critc_target)
        # todo maybe I need to save the learning rate for the adam optimizer
        # self.hard_update()

    def noise_factor_decay(self, step):
        decay_rate = 0.693 / self.params.action_noise_half_decay_time
        self.action_noise_factor = self.params.action_noise_factor * math.exp(-decay_rate * step)
