import threading
import tensorflow as tf
from realips.agent.sac import SacAgent
from realips.trainer.replay_mem import ReplayMemory


class SacTrainerParams:
    def __init__(self):
        self.gamma_discount = 0.99
        self.rm_size = 100000
        self.batch_size = 128
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.0001
        self.is_remote_train = False
        self.actor_freeze_step_count = 5000
        self.use_prioritized_replay = False
        self.pre_fill_exp = 10000
        self.actor_update_period = 5
        self.training_epoch = 32
        self.entropy_factor = 0.5


class SacTrainer:
    def __init__(self, params: SacTrainerParams, agent: SacAgent):
        self.params = params
        self.agent = agent
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_critic)
        self.optimizer_critic_2 = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_critic)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_actor)
        self.replay_mem = ReplayMemory(size=self.params.rm_size)
        self.replay_memory_mutex = threading.Lock()
        self.update_counter = 0

    def store_experience(self, observations, targets, action, reward, next_observations, failed):
        if self.params.is_remote_train:
            self.replay_memory_mutex.acquire()
            self.replay_mem.add((observations, targets, action, reward, next_observations, failed))
            self.replay_memory_mutex.release()
        else:
            self.replay_mem.add((observations, targets, action, reward, next_observations, failed))

    def optimize(self):

        self.update_counter += 1

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

        action_target = self.agent.actor_target([ob2, tgs])
        action_entropy = 0

        q_value_target = tf.minimum(self.agent.critic_target([ob2, tgs, action_target]),
                                    self.agent.critic_target_2([ob2, tgs, action_target]))

        y_exp = r1 + self.params.gamma_discount * q_value_target * (1 - cra)

        with tf.GradientTape() as tape:
            q_value_predicted = self.agent.critic([ob1, tgs, a1])
            loss_critic = tf.keras.losses.mean_squared_error(y_exp, q_value_predicted)
        critic_grads = tape.gradient(loss_critic, self.agent.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(critic_grads, self.agent.critic.trainable_variables))

        with tf.GradientTape() as tape:
            q_value_predicted_2 = self.agent.critic_2([ob1, tgs, a1])
            loss_critic2 = tf.keras.losses.mean_squared_error(y_exp, q_value_predicted_2)
        critic_2_grads = tape.gradient(loss_critic2, self.agent.critic_2.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(critic_2_grads, self.agent.critic_2.trainable_variables))

        # ---------------------- optimize actor ----------------------

        if self.update_counter % self.params.actor_update_period == 0: # update actor less frequently

            if self.replay_mem.get_size() >= self.params.actor_freeze_step_count:
                with tf.GradientTape() as tape:
                    a1_predict = self.agent.actor([ob1, tgs])
                    actor_value = -1 * tf.math.reduce_mean(self.agent.critic([ob1, tgs, a1_predict]))
                actor_gradients = tape.gradient(actor_value, self.agent.actor.trainable_variables)
                self.optimizer_actor.apply_gradients(zip(actor_gradients, self.agent.actor.trainable_variables))

            self.agent.soft_update()