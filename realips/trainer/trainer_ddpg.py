import threading
import tensorflow as tf
from realips.trainer.replay_mem import ReplayMemory
from realips.agent.ddpg import DDPGAgent
from realips.trainer.trainer_params import OffPolicyTrainerParams


class DDPGTrainerParams(OffPolicyTrainerParams):
    def __init__(self):
        super().__init__()
        self.actor_update_period = 1


class DDPGTrainer:
    def __init__(self, params: DDPGTrainerParams, agent: DDPGAgent):
        self.params = params
        self.agent = agent
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_critic)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_actor)
        self.replay_mem = ReplayMemory(size=self.params.rm_size,
                                       combined_experience_replay=self.params.combined_experience_replay)
        self.replay_memory_mutex = threading.Lock()
        self.critic_update = 0

    def store_experience(self, observations, targets, action, reward, next_observations, failed):
        if self.params.is_remote_train:
            self.replay_memory_mutex.acquire()
            self.replay_mem.add((observations, targets, action, reward, next_observations, failed))
            self.replay_memory_mutex.release()
        else:
            self.replay_mem.add((observations, targets, action, reward, next_observations, failed))

    def optimize(self):

        for i in range(self.params.training_epoch):

            if self.params.pre_fill_exp > self.replay_mem.get_size():
                return 0

            self.replay_memory_mutex.acquire()
            mini_batch = self.replay_mem.sample(self.params.batch_size)
            self.replay_memory_mutex.release()

            ob1 = mini_batch[0]
            tgs = mini_batch[1]
            a1 = mini_batch[2]
            r1 = mini_batch[3]
            ob2 = mini_batch[4]
            cra = mini_batch[5]

            # ---------------------- optimize critic ----------------------
            # Use target actor exploitation policy here for loss evaluation

            with tf.GradientTape() as tape:

                a2 = self.agent.actor_target([ob2, tgs])

                if self.params.target_action_noise:
                    action_noise = tf.clip_by_value(
                        tf.random.normal(shape=(self.params.batch_size, 1), mean=0, stddev=0.3),
                        clip_value_min=-0.5, clip_value_max=0.5)

                    a2 = tf.clip_by_value((a2 + action_noise), clip_value_min=-1, clip_value_max=1)

                q_e = self.agent.critic_target([ob2, tgs, a2])

                y_exp = r1 + self.params.gamma_discount * q_e * (1 - cra)
                y_pre = self.agent.critic([ob1, tgs, a1])
                loss_critic = tf.keras.losses.mean_squared_error(y_exp, y_pre)

            q_grads = tape.gradient(loss_critic, self.agent.critic.trainable_variables)
            self.optimizer_critic.apply_gradients(zip(q_grads, self.agent.critic.trainable_variables))

            # ---------------------- optimize actor ----------------------

            self.critic_update += 1

            if self.critic_update % self.params.actor_update_period == 0:
                if self.replay_mem.get_size() >= self.params.actor_freeze_step_count:
                    with tf.GradientTape() as tape:
                        a1_predict = self.agent.actor([ob1, tgs])
                        actor_value = -1 * tf.math.reduce_mean(self.agent.critic([ob1, tgs, a1_predict]))
                    actor_gradients = tape.gradient(actor_value, self.agent.actor.trainable_variables)
                    self.optimizer_actor.apply_gradients(zip(actor_gradients, self.agent.actor.trainable_variables))
            self.agent.soft_update()

            return tf.reduce_mean(loss_critic).numpy()

    def load_weights(self, path):
        self.agent.load_weights(path)
