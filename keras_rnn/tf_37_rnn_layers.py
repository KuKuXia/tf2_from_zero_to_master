import os

import tensorflow as tf
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cell = layers.SimpleRNNCell(3)
cell.build(input_shape=(None, 4))

print(cell.trainable_variables)

# SimpleRNNCell
# out, h1 = call(x, h0)
# x: [b, seq_len, word_vec]
# h0/h1: [b,h dim]
# out： [b, h dim]

x = tf.random.normal([4, 80, 100])
xt0 = x[:, 0, :]

# Single layer RNN cell
cell = layers.SimpleRNNCell(64)
out, xt1 = cell(xt0, [tf.zeros([4, 64])])
print(xt1[0].shape)
print(out.shape)
print(id(out), id(xt1[0]))
print(cell.trainable_variables)

# Multi layer RNN cell
rnn = tf.keras.Sequential([
    layers.SimpleRNN(units=64, dropout=0.5, return_sequences=True, unroll=True),
    layers.SimpleRNN(units=64, dropout=0.5, unroll=True)
])
x = rnn(x)
print(x.shape)
