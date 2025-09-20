import threading
import tensorflow as tf
from realips.agent.sac import AutoSafeSAC
from realips.trainer.replay_mem import ReplayMemory
from realips.trainer.trainer_params import OffPolicyTrainerParams


class AutoSafeTrainerParams(OffPolicyTrainerParams):
    def __init__(self):
        super().__init__()
        self.actor_update_period = 2
        self.step_soft_update = False


class AutoSafeTrainer:
    def __init__(self, params: AutoSafeTrainerParams, agent: AutoSafeSAC):
        self.params = params
        self.agent = agent
        self.replay_mem = ReplayMemory(size=self.params.rm_size,
                                       combined_experience_replay=self.params.combined_experience_replay)
        self.replay_memory_mutex = threading.Lock()
        self.critic_update = 0
        self.clean_buffer = True

    def store_experience(self, observations, action, reward, next_observations, failed):
        if self.params.is_remote_train:
            self.replay_memory_mutex.acquire()
            self.replay_mem.add((observations, action, reward, next_observations, failed))
            self.replay_memory_mutex.release()
        else:
            self.replay_mem.add((observations, action, reward, next_observations, failed))

    def optimize(self):

        if self.clean_buffer and self.replay_mem.get_size() > 300:  # clean the buffer for the manual reset
            self.replay_mem.reset()
            self.clean_buffer = False
            return None

        for i in range(self.params.training_epoch):

            if self.params.pre_fill_exp > self.replay_mem.get_size():
                return None

            self.replay_memory_mutex.acquire()
            mini_batch = self.replay_mem.sample(self.params.batch_size)
            self.replay_memory_mutex.release()

            # ---------------------- optimize critic ----------------------
            training_info = self.agent.optimize(mini_batch)
            # return training_info["critic_loss"], training_info["actor_loss"], training_info["tem"], training_info["lam"]
            return training_info

