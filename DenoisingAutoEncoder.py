"""AutoEncoder."""

import tensorflow as tf
import numpy as np


def xavier_init(fan_in, fan_out, constant=1):
    """初始化权值."""
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low,
                             maxval=high,
                             dtype=tf.float32)
# 在参数中的操作符两侧不能含有spaces


class AdditiveGaussianNoiseAutoencoder(object):
    """去噪自编码器类定义."""

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        """Initialize the class."""
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer_function = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
