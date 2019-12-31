"""
张量限幅
"""

import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.range(10)
print(a)

# clipping
# 保留大于2的元素，小于用2截断
print(tf.maximum(a, 2))
# 保留小于8的元素，大于用8截断
print(tf.minimum(a, 8))
# 保留[2,8]中间的元素
print(tf.clip_by_value(a, 2, 8))

# relu
print('-' * 20)
b = a - 5
print(b)
print(tf.nn.relu(b))
print(tf.maximum(b, 0))

# clip_by_norm
c = tf.random.normal([2, 2], mean=10)
print(c)
print(tf.norm(c))
cc = tf.clip_by_norm(c, 15)
print(cc)
print(tf.norm(cc))
