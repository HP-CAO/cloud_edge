class OffPolicyTrainerParams:
    def __init__(self):
        self.gamma_discount = 0.99
        self.rm_size = 1000000
        self.combined_experience_replay = True
        self.use_prioritized_replay = False
        self.batch_size = 128
        self.learning_rate_actor = 0.0003
        self.learning_rate_critic = 0.0003
        self.is_remote_train = False
        self.actor_freeze_step_count = 5000
        self.pre_fill_exp = 3500
        self.target_action_noise = True
        self.training_epoch = 1

