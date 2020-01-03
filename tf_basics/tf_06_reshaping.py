"""
Transformation
"""

import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Reshape
a = tf.random.normal([4, 28, 28, 3])
print(f'shape of a: {a.shape}, ndim of a: {a.ndim}')

print(tf.reshape(a, [4, 784, 3]).shape)
print(tf.reshape(a, [4, -1, 3]).shape)
print(tf.reshape(a, [4, 784 * 3]).shape)
print(tf.reshape(a, [4, -1]).shape)
# Reshape is flexible
print(tf.reshape(tf.reshape(a, [4, -1]), [4, 28, 28, 3]).shape)
print(tf.reshape(tf.reshape(a, [4, -1, 3]), [-1, 3]).shape)
# Reshape could lead to potential bugs!
# images: [b, h, w, 3] = [ 4, 28, 28, 3] -> [b, pixel, 3]
# 再次转换不一定就是原来的维度了，有很多可能性，如[4, 14, 56, 3]，在处理图像数据时，要记录下转换前数据的对应content，这样才能还原数据


# 对张量数据的content进行调换
b = tf.random.normal((4, 3, 2, 1))
print(f'shape of b: {b.shape}')
print(tf.transpose(b).shape)
print(tf.transpose(b, perm=[0, 1, 3, 2]).shape)

# [b, h, w, 3] -> [b, 3, h, w]
print(tf.transpose(a, [0, 2, 1, 3]).shape)
print(tf.transpose(a, [0, 3, 2, 1]).shape)
print(tf.transpose(a, [0, 3, 1, 2]).shape)

# Squeeze vs Expand_dims
# Expand dim
# c: [classes, students, classes] = [4, 25, 8]
# add school dim
# [1, 4, 25, 8] + [1, 4, 25, 8] = [2, 4, 25, 8]

c = tf.random.normal([4, 35, 8])
print(f'shape of c: {c.shape}')

# Expand_dim： 扩展维度
# 在最后扩展一维
print(tf.expand_dims(c, axis=0).shape)
print(tf.expand_dims(c, axis=-4).shape)

# 在最开始扩展一维
print(tf.expand_dims(c, axis=-1).shape)
print(tf.expand_dims(c, axis=3).shape)

# 在中间扩展一维
print(tf.expand_dims(c, axis=2).shape)
print(tf.expand_dims(c, axis=1).shape)

# Squeeze: 将shape=1的维度删除, axis可以指定要删除的维度，前提是该维度对应的元素是1
# [4, 35, 1, 8] -> [4, 35, 8]
# [4, 1, 35, 1) -> [4, 35]

d = tf.random.normal([1, 2, 5, 1, 1, 3, 1])
print(f'shape of d: {d.shape}')
print(tf.squeeze(d).shape)
print(tf.expand_dims(d, axis=3).shape)
print(tf.expand_dims(d, axis=3).shape)
print(tf.squeeze(d, axis=0).shape)
print(tf.squeeze(d, axis=-1).shape)
print(tf.squeeze(d, axis=4).shape)
