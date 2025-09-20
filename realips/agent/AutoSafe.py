import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Layer
from tensorflow.keras import Model
import numpy as np
import math

ACTION_BOUND = 20.0

F = np.array([[25.9995, 19.4241, 75.9886, 13.8553]]) / ACTION_BOUND # feedback law for the safe controller

MATRIX_P = np.array([[13.3425, 6.73778, 16.2166, 3.47318],
                     [6.73778, 3.94828, 9.69035, 2.09032],
                     [16.2166, 9.69035, 25.9442, 5.31439],
                     [3.47318, 2.09032, 5.31439, 1.16344]])

SAFE_CONTROLLER_STEP = 10  # think of distance|safe value based switching
SAFE_CONTROLLER_ACTIVATE_THRESHOLD = 1.0

SYSTEM_A = np.array([[1.0000, 0.0333, 0, 0],
                     [0.6465, 1.5268, 2.1666, 0.4020],
                     [0, 0, 1.0000, 0.0333],
                     [-1.5151, -1.2348, -4.3123, 0.0577]])

SYSTEM_B = np.array([[0, 0.0334, 0, -0.0783]])

class ModelbasedAgent:
    def __init__(self, feedback_law=F, action_bound=ACTION_BOUND):
        self.feedback_law = feedback_law
        self.matrix_P = MATRIX_P


    def get_action(self, tracking_error):
        # x, x_dot, theta, theta_dot = tracking_error
        # F = np.squeeze(self.feedback_law)
        # action_abs_1 = F[0] * x + F[1] * x_dot + F[2] * theta + F[3] * theta_dot
        action_abs = np.dot(self.feedback_law, tracking_error)
        # Normalize to [-1, 1] # the action bound is determined when calculating the feedback law
        action_abs = np.clip(action_abs, -1, 1)
        # print(f"action_abs: {action_abs}, tracking_error: {tracking_error}")
        return  action_abs

    def safety_switch_on(self, tracking_error):
        safety_value = tracking_error @ MATRIX_P @ tracking_error.T
        return safety_value > 1


class AutoSafeActor(Model):

    def __init__(self, shape_action, P_matrix=MATRIX_P, F_matrix=F, log_std_max=2, log_std_min=-5, tem_min=1.0, tem_max=25.0):
        super(AutoSafeActor, self).__init__()

        self.feature_dense_1 = Dense(256, activation='relu', name='feature_dense_1')
        self.feature_dense_2 = Dense(256, activation='relu', name='feature_dense_2')
        self.feature_dense_3 = Dense(256, activation='relu', name='feature_dense_3')

        self.mean_layer = Dense(shape_action, activation=None, name='mean')
        self.std_layer = Dense(shape_action, activation='tanh', name='std')
        self.temperature_layer = Dense(1, activation='tanh', name='temperature') # [0, 1]  # Here temperature can be a learned vector for each action dimension
        self.P_matrix = P_matrix
        self.F_matrix = F_matrix
        self.n_s, _ = P_matrix.shape
        self.tem_min = tem_min
        self.tem_max = tem_max
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    def call(self, inputs):
        input_state = inputs[:, -self.n_s:]  # Assuming the first n_s dimensions are the state
        input_state_expand = tf.expand_dims(input_state, axis=1)
        x = self.feature_dense_1(inputs)
        x = self.feature_dense_2(x)
        x = self.feature_dense_3(x)
        action_mean_learn = self.mean_layer(x)  # Exclude the last dimension for action mean
        tem =self.temperature_layer(x)  # Adding a small value to avoid division by zero #
        tem = self.tem_min + 0.5 * (self.tem_max - self.tem_min) * (tem + 1)  # Rescale temperature to [TEM_MIN, TEM_MAX]
        log_std_learn = self.std_layer(x)
        log_std_learn = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std_learn + 1)
        input_state_transpose = tf.transpose(input_state_expand, perm=[0, 2, 1])  # Transpose for matrix multiplication
        action_safe = tf.clip_by_value(self.F_matrix @ input_state_transpose, -1, 1)  # Clip the action to [-1, 1]
        action_safe = tf.squeeze(action_safe, axis=-1)  # Remove the extra dimension
        safe_z = input_state_expand @ self.P_matrix @ input_state_transpose
        safe_z = tf.clip_by_value(tf.squeeze(safe_z, axis=1), 0 ,1)
        lam = (tf.math.exp(safe_z * tem) - 1) / (tf.math.exp(1 * tem) - 1)
        return action_mean_learn, log_std_learn, action_safe, lam, tem


class AutoSafeActorSchedule(Model):
    def __init__(self, shape_action, P_matrix=MATRIX_P, F_matrix=F, log_std_max=2, log_std_min=-5, tem_min=1.0, tem_max=25.0,
                 total_steps=500000, schedule_type="linear", tau=5000):
        super(AutoSafeActorSchedule, self).__init__()

        # networks
        self.feature_dense_1 = Dense(256, activation='relu', name='feature_dense_1')
        self.feature_dense_2 = Dense(256, activation='relu', name='feature_dense_2')
        self.feature_dense_3 = Dense(256, activation='relu', name='feature_dense_3')
        self.mean_layer = Dense(shape_action, activation=None, name='mean')
        self.std_layer = Dense(shape_action, activation='tanh', name='std')

        # safe control
        self.P_matrix = tf.constant(P_matrix, dtype=tf.float32)
        self.F_matrix = tf.constant(F_matrix, dtype=tf.float32)
        self.n_s, _ = P_matrix.shape

        # schedule params
        self.tem_min = tem_min
        self.tem_max = tem_max
        self.total_steps = float(total_steps)
        self.schedule_type = schedule_type
        self.tau = float(tau)

        # std clipping
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.tem = tf.Variable(initial_value=1.0)  # minimum value is 1.0

    def update_tem(self, step):
        """Monotonically INCREASING temperature from tem_min -> tem_max."""
        step = tf.cast(step, tf.float32)
        t = tf.clip_by_value(step / self.total_steps, 0.0, 1.0)

        if self.schedule_type == "linear":
            # linear up: tem_min  → tem_max
            tem = self.tem_min + (self.tem_max - self.tem_min) * t
        elif self.schedule_type == "exp":
            # exponential up (fast at start, asymptote to tem_max):
            # tem = tem_min + (tem_max - tem_min) * (1 - exp(-step / tau))
            tem = self.tem_min + (self.tem_max - self.tem_min) * (1.0 - tf.exp(-step / self.tau))
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

        self.tem.assign(tem)

    def call(self, inputs):
        """
        inputs: [B, D]
        step: scalar (int or tf.Tensor)
        """
        input_state = inputs[:, -self.n_s:]
        input_state_expand = tf.expand_dims(input_state, axis=1)

        # learned mean/std
        x = self.feature_dense_1(inputs)
        x = self.feature_dense_2(x)
        x = self.feature_dense_3(x)
        action_mean_learn = self.mean_layer(x)

        log_std_learn = self.std_layer(x)
        log_std_learn = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std_learn + 1.0)

        # safe action
        input_state_transpose = tf.transpose(input_state_expand, perm=[0, 2, 1])
        action_safe = tf.clip_by_value(tf.matmul(self.F_matrix, input_state_transpose), -1, 1)
        action_safe = tf.squeeze(action_safe, axis=-1)

        # scheduled temperature
        tem = self.tem

        # λ computation
        safe_z = input_state_expand @ self.P_matrix @ input_state_transpose
        safe_z = tf.clip_by_value(tf.squeeze(safe_z, axis=1), 0, 1)
        lam = tf.clip_by_value((tf.exp(safe_z * tem) - 1.0) / (tf.exp(1.0 * tem) - 1.0), 0.0, 1.0)

        return action_mean_learn, log_std_learn, action_safe, lam, tem

