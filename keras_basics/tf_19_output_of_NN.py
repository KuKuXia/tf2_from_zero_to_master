"""
输出类型
"""

import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# sigmoid: 只能保证单个点的值范围是[0,1]，不能保证所有的输出值的和为1
a = tf.linspace(-6., 6., 10)  # float number
print(a)
b = tf.sigmoid(a)
print(b)

c = tf.random.normal([1, 28, 28])
print(f"min of c: {tf.reduce_min(c)}, max of c: {tf.reduce_max(c)}")

d = tf.sigmoid(c)
print(f'min of d: {tf.reduce_min(d)}, max of d: {tf.reduce_max(d)}')

a = tf.linspace(-2., 2., 5)
print(tf.sigmoid(a))

# softmax: 保证每个值的范围为[0, 1]，并且所有值的和为1
logits = tf.random.uniform([1, 10], minval=-2, maxval=2)
print(logits)

prob = tf.nn.softmax(logits, axis=1)
print(tf.reduce_sum(prob, axis=1))

# tanh 将值压缩到[-1,1]
print(a)
print(tf.tanh(a))
