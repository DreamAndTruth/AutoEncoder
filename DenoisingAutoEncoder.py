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
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(
            tf.matmul(self.x + self.scale * tf.random_normal(self.n_input),
                      self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                               self.weights['w2']),
                                     self.weights['b2'])
        self.cost = tf.reduce_sum(tf.pow(tf.substruct(self.reconstruction,
                                                      self.x), 2.0))
        self.optimizer = optimizer(self.cost)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(
            xavier_init(self.n_input, self.n_hidden),
            dtype=tf.float32,
            name='w1'
        )
        all_weights['w2'] = tf.Variable(
            xavier_init(self.n_hidden, self.n_input),
            dtype=tf.float32,
            name='w2'
        )
        all_weights['b1'] = tf.Variable(
            tf.zeros([self.n_hidden]),
            dtype=tf.float32,
            name='b1'
        )
        all_weights['b2'] = tf.Variable(
            tf.zeros([self.n_input]),
            dtype=tf.float32,
            name='b2'
        )
        return all_weights

    def partial_fit(self, X):
        """para:X:一个batch_size的数据。执行一步优化."""
        cost, opt = self.sess.run(
            (self.cost, self.optimizer),
            feed_dict={self.x: X, self.scale: self.training_scale}
        )
        return cost

    def calc_total_cost(self, X):
        """test阶段进行cost计算，不进行参数优化."""
        return self.sess.run(
            self.cost,
            feed_dict={self.x: X, self.scale: self.training_scale}
        )

    
