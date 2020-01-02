"""
Tensorflow高阶操作
"""

import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# where:输入是一bool型的

a = tf.random.normal([3, 3])
print(a)
mask = a > 0
print(mask)
a_1 = tf.boolean_mask(a, mask)
print(a_1)
indices = tf.where(mask)
print(indices)
a_2 = tf.gather_nd(a, indices)
print(a_2)

# where(cond, A, B) 如果cond为True，选择A对应的元素，反之选择B对应的元素
A = tf.ones([3, 3])
B = tf.zeros([3, 3])
C = tf.where(mask, A, B)
print(C)

# scatter_nd，底板数据是全零的， 将indices对应位置的数据进行用updates对应的数据替换
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
shape = tf.constant([8])
d = tf.scatter_nd(indices, updates, shape)
print(d)

indices = tf.constant([[0], [2]])
e = tf.constant([[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]])
updates = tf.broadcast_to(e, [2, 4, 4])
print(updates)
shape = tf.constant([4, 4, 4])
f = tf.scatter_nd(indices, updates, shape)
print(f)


# meshgrid

def generate_points_using_numpy():
    points = []
    for y in np.linspace(-2, 2, 5):
        for x in np.linspace(-2, 2, 5):
            points.append([x, y])
    return np.array(points)


def generate_points_using_tf():
    # GPU acceleration
    y = tf.linspace(-2., 2, 5)
    print(y)
    x = tf.linspace(-2., 2, 5)
    print(x)
    points_x, points_y = tf.meshgrid(x, y)
    print(f'shape of point_x: {points_x.shape}, point_y: {points_y.shape}')
    print(points_x)
    print('-' * 30)
    print(points_y)
    points = tf.stack([points_x, points_y], axis=2)
    print(points)
    print(points.shape)


generate_points_using_tf()
