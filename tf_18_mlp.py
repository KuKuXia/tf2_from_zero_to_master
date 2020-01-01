"""
Multi layer perception
"""
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.random.normal([4, 784])

net = tf.keras.layers.Dense(512)
out = net(x)
print(out.shape)
print(net.kernel.shape, net.bias.shape)

# Multi layers using Sequential

x = tf.random.normal([2, 3])
model = tf.keras.Sequential([
    Dense(2, activation='relu'),
    Dense(2, activation='relu'),
    Dense(2)
])

model.build(input_shape=[None, 3])
model.summary()

for p in model.trainable_variables:
    print(p.name, p.shape)
