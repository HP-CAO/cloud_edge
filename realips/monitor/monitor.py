import os
import io
import datetime
import tensorflow as tf
import numpy as np
import shutil
import distutils.util
import matplotlib.pyplot as plt
from realips.env.gym_physics import GymPhysics
from realips.utils import states2observations
from realips.env.reward import RewardFcn


class ModelStatsParams:
    def __init__(self):
        self.max_episode_steps = 1000
        self.total_steps = int(5e6)
        self.target_distance_score = 0.77880078307  # 5 cm distance from the target tape
        self.targets = [0., 0.]  # [x, theta]
        self.model_name = "model_name"
        self.eval_period = 20
        self.log_file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.force_override = False
        self.weights_path = None
        self.running_mode = 'train'
        self.random_initial_ips = True
        self.visualize_eval = False
        self.reset_delay = 1.0
        self.can_swing_up_steps = 100
        self.on_target_reset_steps = 100
        self.converge_episodes = 3
        self.converge_swing_up_time = 250


class ModelStats:

    def __init__(self, params: ModelStatsParams, physics: GymPhysics):
        self.params = params
        self.physics = physics
        self.reward = []
        self.targets = None
        self.observations = None
        self.failed = None
        self.survived = True
        self.distance_scores = []
        self.on_target_steps = 0
        self.consecutive_on_target_steps = 0
        self.cart_positions = []
        self.actions = []
        self.critic_losses = []
        self.pendulum_angele = []
        self.total_steps = 0
        self.converge_eval_episode = 0

        self.log_dir = 'logs/' + self.params.log_file_name
        self.clear_cache()

        self.training_log_writer = tf.summary.create_file_writer(self.log_dir + '/training')
        self.evaluation_log_writer = tf.summary.create_file_writer(self.log_dir + '/eval')

    def get_shape_targets(self):
        return len(self.params.targets)

    def init_episode(self):
        if self.params.random_initial_ips:
            self.physics.random_reset()
            # self.random_set_targets()
            self.targets = self.params.targets
        else:
            self.physics.reset()
            self.targets = self.params.targets

        self.reset_status()
        self.observations, self.failed = states2observations(self.physics.states)

    def reset_status(self):
        self.reward = []
        self.observations = None
        self.failed = False
        self.distance_scores = []
        self.on_target_steps = 0
        self.consecutive_on_target_steps = 0
        self.cart_positions = []
        self.pendulum_angele = []
        self.actions = []
        self.critic_losses = []
        self.survived = True

    def get_average(self, data):
        if len(data) == 0:
            return 0
        else:
            return sum(data) / len(data)

    def get_average_reward(self):
        if len(self.reward) == 0:
            return 0
        else:
            return sum(self.reward) / len(self.reward)

    def get_average_distance_score(self):
        if len(self.distance_scores) == 0:
            return 0
        else:
            return sum(self.distance_scores) / len(self.distance_scores)

    def measure(self, observation, target, crash, pole_length, distance_score_factor):

        distance_score = RewardFcn.get_distance_score(observation, target, pole_length, distance_score_factor)

        if distance_score > self.params.target_distance_score:
            self.on_target_steps += 1
            self.consecutive_on_target_steps += 1
        else:
            self.consecutive_on_target_steps = 0

        self.distance_scores.append(distance_score)

        if crash:
            self.survived = False

    def add_critic_loss(self, loss):
        self.critic_losses.append(loss)

    def get_steps(self):
        return len(self.reward)

    def training_monitor(self, episode):

        average_reward, on_target_steps, average_distance_score, survived, can_swing_up, swing_up_time, critic_loss = self.log_data()
        with self.training_log_writer.as_default():
            tf.summary.scalar('Average_Reward', average_reward, self.total_steps)
            tf.summary.scalar('On_target_step', on_target_steps, self.total_steps)
            tf.summary.scalar('Can_swing_up', can_swing_up, self.total_steps)
            tf.summary.scalar('swing_up_time', swing_up_time, self.total_steps)
            tf.summary.scalar('distance_score', average_distance_score, self.total_steps)
            tf.summary.scalar('critic_loss', critic_loss, self.total_steps)
            tf.summary.scalar('distance_score_and_survived', average_distance_score * survived, self.total_steps)

        print("Training:=====>  Episode: ", episode, " Total steps:",
              self.get_steps(), " Average_reward: ", average_reward, "ds_mean", average_distance_score)

    def evaluation_monitor_scalar(self, episode):

        average_reward, on_target_steps, average_distance_score, survived, can_swing_up, swing_up_time, _ = self.log_data()

        with self.evaluation_log_writer.as_default():
            tf.summary.scalar('Average_Reward', average_reward, self.total_steps)
            tf.summary.scalar('On_target_step', on_target_steps, self.total_steps)
            tf.summary.scalar('Can_swing_up', can_swing_up, self.total_steps)
            tf.summary.scalar('swing_up_time', swing_up_time, self.total_steps)
            tf.summary.scalar('distance_score', average_distance_score, self.total_steps)
            tf.summary.scalar('distance_score_and_survived', average_distance_score * survived, self.total_steps)

        print("Evaluation:=====>  Episode: ", episode, " Total steps:",
              self.get_steps(), " Average_reward: ", average_reward, "ds_mean", average_distance_score)

        if swing_up_time <= self.params.converge_swing_up_time:
            self.converge_eval_episode += 1
        else:
            self.converge_eval_episode = 0

    def evaluation_monitor_image(self, ep):

        average_reward, on_target_steps, average_distance_score, survived, can_swing_up, swing_up_time, _ = self.log_data()
        tf_image = self.plot_to_image(average_reward, on_target_steps, average_distance_score)

        with self.evaluation_log_writer.as_default():
            tf.summary.image('Summary_plot' + str(ep), tf_image, self.total_steps)

    def log_data(self):

        average_reward = self.get_average_reward()
        on_target_steps = self.on_target_steps
        average_distance_score = self.get_average_distance_score()
        survived = self.get_survived()
        can_swing_up = self.consecutive_on_target_steps >= self.params.can_swing_up_steps
        swing_up_time = self.get_steps() - self.consecutive_on_target_steps if can_swing_up else self.params.max_episode_steps
        critic_loss = self.get_average(self.critic_losses)
        return average_reward, on_target_steps, average_distance_score, survived, can_swing_up, swing_up_time, critic_loss

    def random_set_targets(self):
        x_target = np.random.uniform(-self.physics.params.x_threshold, self.physics.params.x_threshold)
        if self.physics.is_failed(x_target, 0):
            x_target = 0.5 * self.physics.params.x_threshold
        self.targets = [x_target, 0]

    def clear_cache(self):
        if os.path.isdir(self.log_dir):
            if self.params.force_override:
                shutil.rmtree(self.log_dir)
            else:
                print(self.log_dir, 'already exists.')
                resp = input('Override log file? [Y/n]\n')
                if resp == '' or distutils.util.strtobool(resp):
                    print('Deleting old log dir')
                    shutil.rmtree(self.log_dir)
                else:
                    print('Okay bye')
                    exit(1)

    def plot_to_image(self, average_reward, on_target_steps, average_distance_score):
        figure = plt.figure()

        x = self.get_steps()
        x_target = (np.zeros(x) + 1) * self.targets[0]
        theta_target = (np.zeros(x) + 1) * self.targets[1]
        plt.subplot(3, 1, 1)
        plt.xlabel('steps')
        plt.ylabel('x/theta')

        plt.plot(np.arange(x), self.cart_positions, label='Cart_position')
        plt.plot(np.arange(x), self.pendulum_angele, label='Pendulum_angle')
        plt.plot(np.arange(x), x_target, '-', linewidth=0.5, label='Position_target')
        plt.plot(np.arange(x), theta_target, '-', linewidth=0.5, label='Angle_target')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.xlabel('steps')
        plt.ylabel('a')

        plt.plot(np.arange(x), self.actions, label='Action')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        tape_x = np.add(np.array(self.cart_positions), np.sin(np.array(self.pendulum_angele)))
        tape_y = np.cos(np.array(self.pendulum_angele))
        label = 'Average_reward: {:.2} \n On_target_steps: ' \
                '{} \n average_distance_score: {:.2}'.format(average_reward, on_target_steps, average_distance_score)

        plt.plot(tape_x[1: -1], tape_y[1: -1], 'o-.k', linewidth=0.5, markeredgewidth=0.3, markeredgecolor='m',
                 markerfacecolor='w',
                 label=label, zorder=1)
        plt.scatter(tape_x[0], tape_y[0], label='Start', marker="*", c='g', s=100, zorder=2)
        plt.scatter(tape_x[-1], tape_y[-1], label='End', marker="*", c='b', s=100, zorder=2)
        plt.scatter(self.targets[0], np.cos(self.targets[1]), label='Target', marker="*", c='r', s=100, zorder=2)

        plt.legend(loc='best', fontsize='x-small')
        plt.grid(True)
        plt.xlim(-2.7, 2.7)
        plt.ylim(-1.5, 1.5)

        figure.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def add_steps(self, step):
        self.total_steps += step

    def get_survived(self):
        return self.survived
