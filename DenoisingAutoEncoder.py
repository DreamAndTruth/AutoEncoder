"""去噪自编码器加性高斯噪声."""
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(fan_in, fan_out, constant=1):
    """生成服从一定分布的矩阵,均值 = 0，方差 = 2/(fan_in+fan_out)."""
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low,
                             maxval=high,
                             dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    """定义去噪自编码器的类."""

    def __init__(self,
                 n_input,
                 n_hidden,
                 transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        """初始化参数."""
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        # 使用tf.placeholder对scale进行参数传递
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 定义自编码器的模型：具有一层隐藏层
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(
            tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                      self.weights['w1']),
            self.weights['b1']))

        # 输出层不使用激活函数
        self.reconstruction = tf.add(
            tf.matmul(self.hidden, self.weights['w2']),
            self.weights['b2'])

        # 定义损失函数的计算方式，使用MSE方法
        self.cost = 0.5 * tf.reduce_sum(
            tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        # 初始化一个tf.Session()会话
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        # 将所有参数组织成一个字典形式
        all_weights = dict()
        all_weights['w1'] = tf.Variable(
            xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(
            tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(
            tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(
            tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        """进行一次优化."""
        cost, opt = self.sess.run(
            (self.cost, self.optimizer),
            feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        """在训练完成的网络中运行，并返回在数据X上的cost，不进行参数更新."""
        return self.sess.run(
            self.cost,
            feed_dict={self.x: X, self.scale: self.training_scale})

    def transform(self, X):
        """返回隐藏层的激活值."""
        return self.sess.run(self.hidden,
                             feed_dict={self.x: X,
                                        self.scale: self.training_scale})

    def generate(self, hidden=None):
        """返回在输入hidden下的重构值."""
        if hidden is None:
            # 生成与b1相同尺寸的矩阵
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        """返回在当前输入X下的网络输出值(与generate进行区分)."""
        return self.sess.run(self.reconstruction,
                             feed_dict={self.x: X,
                                        self.scale: self.training_scale})

    def getWeights(self):
        """返回权值矩阵w1."""
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        """返回偏置矩阵b1."""
        return self.sess.run(self.weights['b1'])


# 读取mnist数据集，数据存放在code当前文件夹下
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def standard_scale(X_train, X_test):
    """使用Sklearn中的类对数据进行标准化处理."""
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    """随机生成一个batch_size数据."""
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index + batch_size)]


X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
# 定义部分训练参数
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1
# 获得一个实例化对象
autoencoder = AdditiveGaussianNoiseAutoencoder(
    n_input=784,
    n_hidden=200,
    transfer_function=tf.nn.softplus,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    scale=0.01)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = autoencoder.partial_fit(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=",
              "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
