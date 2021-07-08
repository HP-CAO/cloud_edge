import tensorflow

def model_summary(self):
    if self.params.log_net_summary:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=100)
        tensorboard_callback.set_model(self.critic)
        tensorboard_callback.set_model(self.actor)
    else:
        return None