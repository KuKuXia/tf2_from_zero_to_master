"""
合并与分割
"""

import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Statistics about scores
# data: [classes, students, scores]
a = tf.ones([4, 35, 8])
b = tf.ones([2., 35, 8])

# Concat连接操作的前提是两个张量在非叠加dim上具有相同的维度
# [class1-4, students, scores]
# [class5-6, students, scores]
c = tf.concat([a, b], axis=0)
print(f'shape of c: {c.shape}')

# [class, student1-35, scores]
# [class, student36-38, scores]
d = tf.ones([4, 3, 8])
print(tf.concat([a, d], axis=1).shape)

# stack：create new dim
# school1: [classes, students, scores]
# school2: [classes, students, scores]
# [schools, classes, students, scores]
e = tf.ones([4, 35, 8])
print(f'shape of a: {a.shape}, e: {e.shape}')
print(tf.concat([a, e], axis=-1).shape)
print(tf.stack([a, e], axis=0).shape)
print(tf.stack([a, e], axis=3).shape)

# Unstack: 根据axis维度解开张量
f = tf.stack([a, e])
print(f'shape of f: {f.shape}')

aa, ee = tf.unstack(f, axis=0)
print(f'shape of aa: {aa.shape}, ee: {ee.shape}')

# 根据第三维度解开张量f，返回一个列表包含8个张量
res = tf.unstack(f, axis=3)
print(f'length of res: {len(res)}, shape of res[0]: {res[0].shape}')

# Split
res_1 = tf.split(f, axis=3, num_or_size_splits=2)
print(f'length of res_1: {len(res_1)}, shape of res[0]: {res_1[0].shape}')

res_2 = tf.split(f, axis=3, num_or_size_splits=[2, 2, 4])
print(
    f'length of res_1: {len(res_2)}, shape of res[0]: {res_2[0].shape}, res[1]: {res_2[1].shape}, res[2]: {res_2[2].shape}')
