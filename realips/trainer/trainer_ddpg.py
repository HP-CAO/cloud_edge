import threading
import tensorflow as tf
from realips.trainer.replay_mem import ReplayMemory
from realips.agent.ddpg import DDPGAgent


class DDPGTrainerParams:
    def __init__(self):
        self.gamma_discount = 0.99
        self.rm_size = 100000
        self.batch_size = 128
        self.learning_rate = 0.0003
        self.is_remote_train = False
        self.actor_freeze_step_count = 5000
        self.use_prioritized_replay = False
        self.pre_fill_exp = 10000


class DDPGTrainer:
    def __init__(self, params: DDPGTrainerParams, agent: DDPGAgent):
        self.params = params
        self.agent = agent
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        self.replay_mem = ReplayMemory(size=self.params.rm_size)

        self.replay_memory_mutex = threading.Lock()

    def store_experience(self, observations, targets, action, reward, next_observations, failed):
        if self.params.is_remote_train:
            self.replay_memory_mutex.acquire()
            self.replay_mem.add((observations, targets, action, reward, next_observations, failed))
            self.replay_memory_mutex.release()
        else:
            self.replay_mem.add((observations, targets, action, reward, next_observations, failed))

    def optimize(self):

        if self.params.use_prioritized_replay:
            self.optimize_prioritized()
            return

        if self.params.pre_fill_exp > self.replay_mem.get_size():
            return

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
            q_e = self.agent.critic_target([ob2, tgs, a2])
            y_exp = r1 + self.params.gamma_discount * q_e * (1 - cra)
            y_pre = self.agent.critic([ob1, tgs, a1])
            loss_critic = tf.keras.losses.mean_squared_error(y_exp, y_pre)
            q_grads = tape.gradient(loss_critic, self.agent.critic.trainable_variables)
            self.optimizer_critic.apply_gradients(zip(q_grads, self.agent.critic.trainable_variables))

        # ---------------------- optimize actor ----------------------
        if self.replay_mem.get_size() >= self.params.actor_freeze_step_count:
            with tf.GradientTape() as tape:
                a1_predict = self.agent.actor([ob1, tgs])
                actor_value = -1 * tf.math.reduce_mean(self.agent.critic([ob1, tgs, a1_predict]))
                actor_gradients = tape.gradient(actor_value, self.agent.actor.trainable_variables)
                self.optimizer_actor.apply_gradients(zip(actor_gradients, self.agent.actor.trainable_variables))

        self.agent.soft_update()

    def optimize_prioritized(self):

        if self.params.batch_size > self.replay_mem.get_size():
            return

        self.replay_memory_mutex.acquire()
        idx, mini_batch, importance_sampling_weight = self.replay_mem.sample_prioritized(self.params.batch_size)
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
            q_e = self.agent.critic_target([ob2, tgs, a2])
            y_exp = r1 + self.params.gamma_discount * q_e * (1 - cra)
            y_pre = self.agent.critic([ob1, tgs, a1])
            td_error = y_exp - y_pre
            loss_critic = tf.reduce_mean(importance_sampling_weight * td_error ** 2)
            q_grads = tape.gradient(loss_critic, self.agent.critic.trainable_variables)
            # q_grads = importance_sampling_weight * td_error * bare_grads
            self.optimizer_critic.apply_gradients(zip(q_grads, self.agent.critic.trainable_variables))
            self.replay_mem.update_priority(idx, abs(td_error.numpy()))

        # ---------------------- optimize actor ----------------------
        if self.replay_mem.get_size() >= self.params.actor_freeze_step_count:
            with tf.GradientTape() as tape:
                a1_predict = self.agent.actor([ob1, tgs])
                actor_value = -1 * tf.math.reduce_mean(self.agent.critic([ob1, tgs, a1_predict]))
                actor_gradients = tape.gradient(actor_value, self.agent.actor.trainable_variables)
                self.optimizer_actor.apply_gradients(zip(actor_gradients, self.agent.actor.trainable_variables))

        self.agent.soft_update()

    def load_weights(self, path):
        self.agent.load_weights(path)
