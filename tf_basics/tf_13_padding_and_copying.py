"""
填充与复制
"""
import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# padding
a = tf.reshape(tf.range(9), [3, 3])
print(a)
print(tf.pad(a, [[0, 0], [0, 0]]))
print(tf.pad(a, [[0, 0], [0, 0]]))
print(tf.pad(a, [[1, 1], [0, 0]]))
print(tf.pad(a, [[1, 1], [1, 0]]))
print(tf.pad(a, [[1, 1], [1, 1]]))

# image padding
b = tf.random.normal([4, 28, 28, 3])
c = tf.pad(b, [[0, 0], [2, 2], [2, 2], [0, 0]])
print(c.shape)

# tile, repeat data along dim n times
# [a, b, c], 2 -> [a,b,c,a,b,c]
# inner dim first
print(a)
print(a.shape)
print(tf.tile(a, [1, 2]))
print(tf.tile(a, [2, 1]))
print(tf.tile(a, [2, 2]))

# tile vs broadcast
aa = tf.expand_dims(a, axis=0)  # -> [1,3,3]
print(aa.shape)
print(tf.tile(aa, [2, 1, 1]))
print(tf.broadcast_to(a, [2, 3, 3]))
