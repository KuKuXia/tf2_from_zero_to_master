"""
CNN
"""

import os

import tensorflow as tf
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.random.normal([10, 32, 32, 3])
layer = layers.Conv2D(4, kernel_size=5, strides=1, padding='valid')
out = layer(x)
print(out.shape)

layer = layers.Conv2D(4, kernel_size=5, strides=1, padding='same')
out = layer(x)
print(out.shape)

layer = layers.Conv2D(20, kernel_size=5, strides=2, padding='same')
out = layer(x)
print(out.shape)
print(layer.call(x).shape)

# 查看weight和bias的shape
print(layer.kernel.shape)
print(layer.bias.shape)

# 调用function版本的Conv2d操作，不常用
w = tf.random.normal([5, 5, 3, 4])
b = tf.zeros([4])
print(x.shape)

out = tf.nn.conv2d(x, w, strides=1, padding='VALID')
print(out.shape)
out = out + b
print(out.shape)
out = tf.nn.conv2d(x, w, strides=2, padding='VALID')
print(out.shape)

# Pooling: 下采样
print('-' * 30)
x = tf.random.normal([1, 14, 14, 4])

pool = layers.MaxPooling2D(2, strides=2)
out = pool(x)
print(out.shape)

pool = layers.MaxPooling2D(3, strides=2)
out = pool(x)
print(out.shape)

# 调用functional版本的max_pool2d操作，不常用
out = tf.nn.max_pool2d(x, 2, strides=2, padding='VALID')
print(out.shape)

# Upsampling: 上采样
print('-' * 30)
x = tf.random.normal([1, 7, 7, 4])

layer = layers.UpSampling2D(size=3)
out = layer(x)
print(out.shape)

layer = layers.UpSampling2D(size=2)
out = layer(x)
print(out.shape)

# Relu: 过滤掉数据中的负值
x = tf.random.normal([2, 3])
print(x)
print(tf.nn.relu(x))
print(layers.ReLU()(x))
