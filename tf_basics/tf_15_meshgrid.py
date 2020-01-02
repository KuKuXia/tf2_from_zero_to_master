import os

import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def func(x):
    """

    :param x: [b, 2]
    :return:
    """
    z = tf.math.sin(x[..., 0]) + tf.math.sin(x[..., 1])

    return z


x = tf.linspace(0., 2 * 3.14, 500)
y = tf.linspace(0., 2 * 3.14, 500)
print(f'shape of x: {x.shape}, y: {y.shape}')

# [500, 500]
point_x, point_y = tf.meshgrid(x, y)
print(f'shape of x: {point_x.shape}, y: {point_y.shape}')

# [500, 500, 2]
points = tf.stack([point_x, point_y], axis=2)
# points = tf.reshape(points, [-1, 2])
print('points:', points.shape)
z = func(points)
print('z:', z.shape)

plt.figure('plot 2d func value')
plt.imshow(z, origin='lower', interpolation='none')
plt.colorbar()

# 等高线描绘
plt.figure('plot 2d func contour')
plt.contour(point_x, point_y, z)
plt.colorbar()
plt.show()
