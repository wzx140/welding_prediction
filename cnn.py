import tensorflow as tf
import numpy as np


class Cnn(object):

    def __init__(self, conv_layers, fc_layers, filters, learning_rate):
        self.__param = {}
        self.__conv_layers = conv_layers
        self.__fc_layers = fc_layers
        self.__filters = filters
        self.__lr = learning_rate

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    def initialize(self, n_w0, n_c0, n_y):
        """
        initialize the w, X, Y
        :param n_w0: width of the image data
        :param n_c0: depth of the image data
        :param n_y: number of the labels
        :return:
        """
        f = self.__filters
        convs = self.__conv_layers
        self.__x = tf.placeholder(tf.float32, (None, n_w0, n_c0), 'data_x')
        self.__y = tf.placeholder(tf.float32, (None, n_y), 'data_y')

        # i->index of layers include pool, temp_i->index of layers exclude pool
        temp_i = 0

        for i in range(len(convs)):
            if i == 0:
                self.__param['w1'] = tf.get_variable('variable_w1', (f[i][0], n_c0, convs[i]),
                                                     initializer=tf.contrib.layers.xavier_initializer())
                temp_i += 1
            elif convs[i] != 0 and convs[i] != -1:
                self.__param['w' + str(temp_i + 1)] = tf.get_variable('variable_w' + str(temp_i + 1), (
                    f[i][0], self.__param['w' + str(temp_i)].shape[2], convs[i]),
                                                                      initializer=tf.contrib.layers.xavier_initializer())
                temp_i += 1
                # todo: add b
                # self.__param['b' + str(i + 1)] = tf.get_variable('b' + str(i + 1),(1,1,3,convs[i]),initializer= )

    def forward(self, drop_out=False):
        param = self.__param
        f = self.__filters
        convs = self.__conv_layers
        fcs = self.__fc_layers
        a_pre = self.__x

        # i->index of layers include pool, temp_i->index of layers exclude pool
        temp_i = 0

        # convolution forward
        for i in range(len(convs)):
            if convs[i] == 0:
                a = tf.nn.pool(a_pre, window_shape=[f[i][0]], padding=f[i][2], pooling_type="AVG",
                               name='pool_a' + str(temp_i))
                a_pre = a
            elif convs[i] == -1:
                a = tf.nn.pool(a_pre, window_shape=[f[i][0]], pooling_type="MAX",
                               padding=f[i][2], name='pool_a' + str(temp_i))
                a_pre = a
            else:
                z = tf.nn.conv1d(a_pre, param['w' + str(temp_i + 1)], stride=f[i][1], padding=f[i][2],
                                 name='z' + str(i))
                a = tf.nn.relu(z, name='a' + str(i))
                a_pre = a
                temp_i += 1

        # full connected forward
        a_pre = tf.contrib.layers.flatten(a)
        for i in range(len(fcs)):
            a = tf.contrib.layers.fully_connected(a_pre, fcs[i])
            a_pre = a
        a = tf.contrib.layers.fully_connected(a_pre, 1, activation_fn=None)
        self.__a = a

    def cost(self):
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.__a, labels=self.__y))
        return cost

    def get_optimizer(self, cost):
        adam = tf.train.AdamOptimizer(self.__lr).minimize(cost)
        return adam

    def predict(self):
        # self.forward(drop_out=False)
        pre = tf.cast(tf.greater(self.__a, 0.5), dtype=np.float, name='output')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pre, self.__y), "float"))
        return pre, accuracy
