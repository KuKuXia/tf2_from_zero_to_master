"""
Broadcasting: 对张量的维度进行扩展的手段，指对某一个维度重复n多次，但是并没有真正的复制数据
tf.tile:对张量某个维度进行显式的复制n多次数据，并且会真实地在数据上体现出来
"""
import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Broadcasting
a = tf.random.normal([4, 32, 32, 3])
print(f'shape of a: {a.shape}')
print((a + tf.random.normal([3])).shape)
print((a + tf.random.normal([32, 32, 1])).shape)
print((a + tf.random.normal([4, 1, 1, 1])).shape)
# print((a + tf.random.normal([1, 4, 1, 1])).shape)  # 报错，因为第2维度有4个元素，和32不一致


# tf.broadcast_to
# 将张量通过broadcast变为另一个shape
original_b = tf.random.normal([4, 1, 1, 1])
after_b = tf.broadcast_to(original_b, [4, 32, 32, 3])
print(f'original shape of b: {original_b.shape}, after broadcasting, b: {after_b.shape}')

# tf.tile
c = tf.ones([3, 4])
c_1 = tf.broadcast_to(c, [2, 3, 4])
print(f'shape of c: {c.shape}, c_1: {c_1.shape}')

c_2 = tf.expand_dims(c, axis=0)
c_3 = tf.tile(c_2, [2, 1, 1])  # 表示在特定的维度复制多少次
print(f'shape of c_2: {c_2.shape}, c_3: {c_3.shape}')
