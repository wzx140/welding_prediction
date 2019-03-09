from enum import Enum, unique

import tensorflow as tf


@unique
class REGULARIZATION(Enum):
    L2 = 1
    DROPOUT = 2


class Cnn(object):
    """
    deep neural network. This class just build the calculation graph. You still need to run with tf session
    """

    def __init__(self, layer_dims: list, mini_batch_size: int, learning_rate: float, keep_prob: float = 1,
                 lambd: float = 0, beta1: float = 0.9, beta2: float = 0.999):
        """
        This can not happen:
            1. keep_prob != 1 and lambd != 0
            2. keep_prob > 1
        :param layer_dims: the dimensions of each layer in dnn
        :param mini_batch_size: the size of mini batch in MBGD.
        :param beta1: use in adam
        :param beta2: use in adam
        """
        # store w and b
        self.__param = {}

        self.__lr = learning_rate
        self.__size = mini_batch_size
        self.__layer_dims = layer_dims
        self.__beta1 = beta1
        self.__beta2 = beta2

        # the output of the network
        self.__a = None

        # input and output value
        self.__x = tf.placeholder(tf.float32, (layer_dims[0], None), 'X')
        self.__y = tf.placeholder(tf.float32, (1, None), 'Y')

        # operate regularization
        assert (0 <= keep_prob <= 1)
        assert (keep_prob == 1 or lambd == 0)
        if keep_prob != 1:
            self.__regularization = REGULARIZATION.DROPOUT
            self.__keep_prob = keep_prob
        elif lambd != 0:
            self.__regularization = REGULARIZATION.L2
            self.__lambd = lambd
        else:
            self.__regularization = None

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    def initialize_parameters(self):
        layers = self.__layer_dims

        for l in range(1, len(layers)):
            self.__param['W' + str(l)] = tf.get_variable(name='W' + str(l),
                                                         shape=(self.__layer_dims[l], self.__layer_dims[l - 1]),
                                                         dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer())
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.__param['W' + str(l)])
            self.__param['b' + str(l)] = tf.get_variable(name='b' + str(l), shape=(self.__layer_dims[l], 1),
                                                         dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer())

    def forward(self, dropout=True):
        """
        forward propagation
        :param dropout
        """
        layers = self.__layer_dims
        param = self.__param

        a_pre = self.__x

        # First L-1 layers use relu function
        for l in range(1, len(layers) - 1):
            z = tf.matmul(param['W' + str(l)], a_pre, name='Z' + str(l)) + param['b' + str(l)]
            a = tf.nn.relu(z, name='A' + str(l))
            if self.__regularization == REGULARIZATION.DROPOUT and dropout:
                a = tf.nn.dropout(a, self.__keep_prob)
            a_pre = a

        # last layer use sigmoid function
        z = tf.matmul(param['W' + str(l + 1)], a_pre, name='Z' + str(l + 1)) + param['b' + str(l + 1)]
        a = tf.nn.sigmoid(z, name='A' + str(l + 1))
        self.__a = a

    def cost(self):
        """
        compute the cost
        :return:
        """
        if self.__regularization == REGULARIZATION.L2:
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.__lambd / self.__size, )
            reg = tf.contrib.layers.apply_regularization(regularizer)
            cost = tf.reduce_mean(tf.square(self.__a - self.__y)) + reg
        else:
            # Mean Squared Error
            cost = tf.reduce_mean(tf.square(self.__a - self.__y))

        return cost

    def get_optimizer(self, cost):
        adam = tf.train.AdamOptimizer(self.__lr, beta1=self.__beta1, beta2=self.__beta2).minimize(cost)
        return adam

    def predict(self):
        self.forward(dropout=False)
        pre = tf.cast(tf.greater(self.__a, 0.5), tf.float32, name='output')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pre, self.__y), tf.float32))
        return self.__a, accuracy
