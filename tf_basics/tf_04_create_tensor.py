"""
创建张量
"""
import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 利用numpy，list创建张量
# 创建全为1的张量
a = tf.convert_to_tensor(np.ones([3, 3]))
a_1 = tf.convert_to_tensor(np.ones([]))
a_2 = tf.convert_to_tensor(np.ones([2]))

# tf.zeros: 创建零张量
b = tf.convert_to_tensor(np.zeros([2, 3]))  # 传入的是数据的shape，而不是data
b_like_1 = tf.convert_to_tensor(np.zeros_like(a))  # 创建一个和a一样shape的零张量
b_like_2 = tf.convert_to_tensor(np.zeros(a.shape))  # 同上的功能

# 利用列表作为data创建张量
c = tf.convert_to_tensor([1, 2])
d = tf.convert_to_tensor([1, 2.])
e = tf.convert_to_tensor([[1], [2.]]).gpu()  # 创建一个在gpu上的张量

print(f'a: {a}')
print(f'a_1: {a_1}')
print(f'a_2: {a_2}')
print(f'b: {b}')
print(f'b: {b_like_1}')
print(f'b: {b_like_2}')
print(f'c: {c}')
print(f'd: {d}')
print(f'e: {e}')

# 利用tf直接创建
# tf.zeros创建零张量
a = np.ones([2, 3])
tf_a = tf.zeros([])
tf_a_1 = tf.zeros([1])
tf_a_2 = tf.zeros([1, 2])
tf_a_3 = tf.zeros(a.shape)  # 创建和a一样shape的零张量
tf_a_4 = tf.zeros_like(a)  # 创建和a一样shape的零张量

print(f'tf_a: {tf_a}')
print(f'tf_a_1: {tf_a_1}')
print(f'tf_a_2: {tf_a_2}')
print(f'tf_a_3: {tf_a_3}')
print(f'tf_a_4: {tf_a_4}')

# tf.ones创建1张量
b = np.ones([3, 3])
tf_b = tf.ones(1)
tf_b_1 = tf.ones([])
tf_b_2 = tf.ones([3, 3])
tf_b_3 = tf.ones(b.shape)  # 创建和a一样shape的1张量
tf_b_4 = tf.ones_like(b)  # 创建和a一样shape的1张量

print(f'tf_b: {tf_b}')
print(f'tf_b_1: {tf_b_1}')
print(f'tf_b_2: {tf_b_2}')
print(f'tf_b_3: {tf_b_3}')
print(f'tf_b_4: {tf_b_4}')

# tf.fill创建元素全为某个数的张量
c = tf.fill([2, 2], 0)
c_1 = tf.fill([2, 2], 1)
c_2 = tf.fill([2, 2], 9)
print(f'c: {c}')
print(f'c_1: {c_1}')
print(f'c_2: {c_2}')

# Random Tensor
# Normal：正态分布
d = tf.random.normal([2, 2], mean=1, stddev=1)
d_1 = tf.random.normal([2, 2])  # 默认为均值0，方差1
d_2 = tf.random.truncated_normal([2, 2], mean=0, stddev=1)  # 截断分布，因为在神经网络中，存在梯度消散的现象，将正态分布的左右两边截断，靠近中间采样

print(f'd: {d}')
print(f'd_1: {d_1}')
print(f'd_2: {d_2}')

# 均匀分布
e = tf.random.uniform([2, 2], minval=0, maxval=100)
print(f'e: {e}')

# Application
# Random Permutation
# [10, 28, 28, 3]: 图片数据，按照第一维度对64进行随机排序
idx = tf.range(10)
print('Original idx: ', idx)
idx = tf.random.shuffle(idx)
print('Shuffled idx: ', idx)

a = tf.random.normal([10, 784])
b = tf.random.uniform([10], maxval=10, dtype=tf.int32)
print(f'Original a: {a}')
print(f'Original b: {b}')

# Random Permutation
a_shuffled = tf.gather(a, idx)
b_shuffled = tf.gather(b, idx)
print(f'a shuffled: ', a_shuffled)
print(f'b shuffled: ', b_shuffled)

# Loss
# 随机产生4个带有10个值的数据作为输出
out = tf.random.uniform([4, 10])
# 随机产生4个准确的label
y = tf.range(4)
# 对label进行one_hot编码
y = tf.one_hot(y, depth=10)
# 计算输出和label的mse值
loss = tf.keras.losses.mse(y, out)
print(f'loss of each sample: {loss}')
loss = tf.reduce_mean(loss)
print(f'averaged loss: {loss}')
