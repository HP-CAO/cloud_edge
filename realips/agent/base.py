import math
import tensorflow as tf
import os
from keras_flops import get_flops
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from realips.utils import OrnsteinUhlenbeckActionNoise
import numpy as np


class BaseAgentParams:
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


class BaseAgent:
    def __init__(self, params: BaseAgentParams, shape_observations=5, shape_targets=2, shape_action=1,
                 on_edge=False):
        self.params = params
        self.shape_observations = shape_observations
        self.shape_targets = shape_targets
        self.shape_action = shape_action
        self.on_edge = on_edge
        self.action_noise = OrnsteinUhlenbeckActionNoise(self.shape_action)
        self.action_noise_factor = params.action_noise_factor
        self.add_actions_observations = self.params.add_actions_observations
        self.action_observations_dim = self.params.action_observations_dim
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None

    def initial_model(self):
        self.actor = self.build_actor("normal-")

    def load_weights(self, path_to_weights):
        print("loading pretrained weights......")

        if not os.path.exists(path_to_weights):
            raise IOError("Weights path not exist")

        path_to_actor = path_to_weights + 'actor_weights'

        self.actor.load_weights(path_to_actor)

    def build_critic(self, name):
        """
        return: Model of critic neural network
        """
        # observations input branch
        input_observations = Input(shape=(self.shape_observations,), name=name + 'observations_input',
                                   dtype=tf.float16)
        input_targets = Input(shape=(self.shape_targets,), name=name + 'targets_input', dtype=tf.float16)
        input_action = Input(shape=(self.shape_action,), name=name + 'action_input', dtype=tf.float16)

        input_concat = tf.concat([input_observations, input_targets, input_action], 1, name=name + 'input_concat')

        dense1_critic = Dense(256, activation='relu', name=name + 'dense1_critic')(input_concat)
        dense2_critic = Dense(128, activation='relu', name=name + 'dense2_critic')(dense1_critic)
        dense3_critic = Dense(64, activation='relu', name=name + 'dense3_critic')(dense2_critic)
        output_value = Dense(1, activation=None, name=name + 'output_critic')(dense3_critic)

        # generate critic network
        model_critic = Model(inputs=[input_observations, input_targets, input_action], outputs=output_value,
                             name=name + 'critic')

        return model_critic

    def build_actor(self, name):
        """
        return: model of actor neural network
        """
        input_observations = Input(shape=(self.shape_observations,), name=name + 'observations_input',
                                   dtype=tf.float16)
        input_targets = Input(shape=(self.shape_targets,), name=name + 'targets_input', dtype=tf.float16)
        input_concat = tf.concat([input_observations, input_targets], 1, name=name + 'input_concat')

        dense1_actor = Dense(self.params.actor_dense1, activation='relu', name=name + 'dense1_actor')(input_concat)
        dense2_actor = Dense(self.params.actor_dense2, activation='relu', name=name + 'dense2_actor')(dense1_actor)
        dense3_actor = Dense(self.params.actor_dense3, activation='relu', name=name + 'dense3_actor')(dense2_actor)
        output_action = Dense(self.shape_action, activation='tanh', name=name + 'action_output')(dense3_actor)

        # generate actor network
        model_actor = Model(inputs=[input_observations, input_targets], outputs=output_action, name=name + 'actor')
        # flops = get_flops(model_actor, batch_size=1)
        # print(f"FLOPS: {flops / 10 ** 9:.03} G")
        # print("analysis")
        return model_actor

    def get_exploration_action(self, observations, targets):
        """
        get the action from actor and add exploration noise
        :param targets: a list
        :param observations: a list
        :return: a scalar value
        """
        var = self.action_noise_factor
        observations = tf.expand_dims(observations, 0)  # add batch dim
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
        observations = tf.expand_dims(observations, 0)  # add batch dim
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

    def get_actor_weights(self):
        return self.actor.get_weights()

    def set_actor_weights(self, weights):
        self.actor.set_weights(weights)

    def noise_factor_decay(self, step):
        decay_rate = 0.693 / self.params.action_noise_half_decay_time
        self.action_noise_factor = self.params.action_noise_factor * math.exp(-decay_rate * step)

    def set_action_noise_factor(self, action_noise_factor):
        self.action_noise_factor = action_noise_factor
