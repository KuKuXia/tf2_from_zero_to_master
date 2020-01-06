import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(22)
np.random.seed(22)

batch_size = 128

# the most frequent words
total_words = 10000
max_review_len = 80
embedding_len = 100

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batch_size, drop_remainder=True)  # 最后一个batch可能不是刚刚好batch_size大小，因此丢掉

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batch_size, drop_remainder=True)  # 最后一个batch可能不是刚刚好batch_size大小，因此丢掉

sample = next(iter(db_train))
print('x_train shape: ', sample[0].shape, tf.reduce_max(sample[0]), tf.reduce_min(sample[0]))
print('y_train shape: ', sample[1].shape, tf.reduce_max(sample[1]), tf.reduce_min(sample[1]))


class RNNUsingSimpleRNNCell(keras.Model):
    def __init__(self, units):
        super(RNNUsingSimpleRNNCell, self).__init__()

        # [b, 64]
        self.state0 = [tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units])]
        self.state2 = [tf.zeros([batch_size, units])]
        self.state3 = [tf.zeros([batch_size, units])]

        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)

        # [b, 80. 100], h_dim: 64
        # RNN: cell, cell2, cell3
        # SimpleRNN
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell2 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell3 = layers.SimpleRNNCell(units, dropout=0.5)

        # fc, [b, 80, 100] => [b, 64] => [b, 1]
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        net(x) net(x, training=True): train mode
        net(x, training=False): test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # [b, 80, 100] => [b, 64]
        state0 = self.state0
        state1 = self.state1
        state2 = self.state2
        state3 = self.state3
        out3 = None
        for word in tf.unstack(x, axis=1):  # word: [b, 100]
            # h1 = x*wxh + h0*whh
            # out0: [b, 64]
            out0, state0 = self.rnn_cell0(word, state0, training)
            # out1: [b, 64]
            out1, state1 = self.rnn_cell1(out0, state1, training)
            out2, state2 = self.rnn_cell1(out1, state2, training)
            out3, state3 = self.rnn_cell1(out2, state3, training)

        # out: [b, 64] => [b, 1]
        x = self.output_layer(out3)

        # p(y is pos|s)
        prob = tf.sigmoid(x)
        return prob


class RNNUsingSimpleRNN(keras.Model):

    def __init__(self, units):
        super(RNNUsingSimpleRNN, self).__init__()

        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)

        # [b, 80, 100] , h_dim: 64
        self.rnn = keras.Sequential([
            layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.SimpleRNN(units, dropout=0.5, unroll=True)
        ])

        # fc, [b, 80, 100] => [b, 64] => [b, 1]
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        net(x) net(x, training=True) :train mode
        net(x, training=False): test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # x: [b, 80, 100] => [b, 64]
        x = self.rnn(x, training=training)

        # out: [b, 64] => [b, 1]
        x = self.output_layer(x)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob


class LSTMUsingLSTMCell(keras.Model):
    def __init__(self, units):
        super(LSTMUsingLSTMCell, self).__init__()

        # [b, 64]
        self.state0 = [tf.zeros([batch_size, units]), tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units]), tf.zeros([batch_size, units])]

        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)

        # [b, 80, 100] , h_dim: 64
        # RNN: cell1 ,cell2, cell3
        # SimpleRNN
        self.rnn_cell0 = layers.LSTMCell(units, dropout=0.5)
        self.rnn_cell1 = layers.LSTMCell(units, dropout=0.5)

        # fc, [b, 80, 100] => [b, 64] => [b, 1]
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        net(x) net(x, training=True) :train mode
        net(x, training=False): test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # [b, 80, 100] => [b, 64]
        state0 = self.state0
        state1 = self.state1
        out1 = None
        for word in tf.unstack(x, axis=1):  # word: [b, 100]
            # h1 = x*wxh+h0*whh
            # out0: [b, 64]
            out0, state0 = self.rnn_cell0(word, state0, training)
            # out1: [b, 64]
            out1, state1 = self.rnn_cell1(out0, state1, training)

        # out: [b, 64] => [b, 1]
        x = self.output_layer(out1)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob


class LSTMUsingLSTM(keras.Model):

    def __init__(self, units):
        super(LSTMUsingLSTM, self).__init__()

        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)

        # [b, 80, 100] , h_dim: 64
        self.rnn = keras.Sequential([
            layers.LSTM(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.LSTM(units, dropout=0.5, unroll=True)
        ])

        # fc, [b, 80, 100] => [b, 64] => [b, 1]
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        net(x) net(x, training=True) :train mode
        net(x, training=False): test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # x: [b, 80, 100] => [b, 64]
        x = self.rnn(x, training=training)

        # out: [b, 64] => [b, 1]
        x = self.output_layer(x)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob


class GRUUsingGRUCell(keras.Model):

    def __init__(self, units):
        super(GRUUsingGRUCell, self).__init__()

        # [b, 64]
        self.state0 = [tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units])]

        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)

        # [b, 80, 100] , h_dim: 64
        # RNN: cell1 ,cell2, cell3
        # SimpleRNN
        self.rnn_cell0 = layers.GRUCell(units, dropout=0.5)
        self.rnn_cell1 = layers.GRUCell(units, dropout=0.5)

        # fc, [b, 80, 100] => [b, 64] => [b, 1]
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        net(x) net(x, training=True) :train mode
        net(x, training=False): test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # [b, 80, 100] => [b, 64]
        state0 = self.state0
        state1 = self.state1
        out1 = None
        for word in tf.unstack(x, axis=1):  # word: [b, 100]
            # h1 = x*wxh+h0*whh
            # out0: [b, 64]
            out0, state0 = self.rnn_cell0(word, state0, training)
            # out1: [b, 64]
            out1, state1 = self.rnn_cell1(out0, state1, training)

        # out: [b, 64] => [b, 1]
        x = self.output_layer(out1)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob


class GRUUsingGRU(keras.Model):

    def __init__(self, units):
        super(GRUUsingGRU, self).__init__()

        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)

        # [b, 80, 100] , h_dim: 64
        self.rnn = keras.Sequential([
            # unroll: Boolean (default False). If True, the network will be unrolled,
            # else a symbolic loop will be used.
            # Unrolling can speed-up a RNN, although it tends to be more memory-intensive.
            # Unrolling is only suitable for short sequences.
            layers.GRU(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.GRU(units, dropout=0.5, unroll=True)
        ])

        # fc, [b, 80, 100] => [b, 64] => [b, 1]
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        net(x) net(x, training=True) :train mode
        net(x, training=False): test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # x: [b, 80, 100] => [b, 64]
        x = self.rnn(x, training=training)

        # out: [b, 64] => [b, 1]
        x = self.output_layer(x)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob


def main():
    units = 64
    epochs = 10
    model = LSTMUsingLSTMCell(units)
    model.compile(optimizers=keras.optimizers.Adam(0.001), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'],
                  experimental_run_tf_function=False)
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    model.summary()
    model.evaluate(db_test)


if __name__ == '__main__':
    main()
