import tensorflow as tf

a = tf.ones([1, 5, 5, 3])
print(a[0][0])
print(a[0][0][0])
print(a[0][0][0][2])

# numpy-style indexing
print('-' * 20)
a = tf.random.normal([4, 28, 28, 3])
print(a[1].shape)
print(a[1, 2].shape)
print(a[1, 2, 3].shape)
print(a[1, 2, 3, 2].shape)

# start:end
print('-' * 20)
b = tf.range(10, dtype=tf.float32)
print(b[-1:])
print(b[4:-1])
print(b[-2:])

# indexing by :
print('-' * 20)
print(a.shape)
print(a[0, :, :, :].shape)
print(a[0, 1, :, :].shape)
print(a[:, :, :, 0].shape)
print(a[:, :, 2, :].shape)
print(a[:, 0, :, :].shape)

# indexing by ::
# start:end:step
# ::step
print('-' * 20)
print(a.shape)
print(a[0:2, :, :, :].shape)
print(a[:, 0:28:2, 0:28:2, :].shape)
print(a[:, :14, :14, :].shape)
print(a[:, 14:, 14:, :].shape)
print(a[:, ::3, ::3, :].shape)

# ::-1
a = tf.range(4)
print(a)
print(a[::-1])
print(a[::-2])
print(a[2::-2])

# ...
print('-' * 20)
a = tf.random.normal([2, 4, 28, 28, 3])
print(a[0].shape)
print(a[0, :, :, :, :].shape)
print(a[0, ...].shape)
print(a[:, :, :, :, 0].shape)
print(a[..., 0].shape)
print(a[0, ..., 2].shape)
print(a[1, 0, ..., 0].shape)

# Selective indexing
# tf.gather: 代表从单个axis上挑选位于index的数据
print('-' * 20)
# data: [classes, students, subjects]
data = tf.random.normal([4, 35, 8])
print(tf.gather(data, axis=0, indices=[2, 3]).shape)
print(data[2:4].shape)
print(tf.gather(data, axis=0, indices=[2, 1, 3, 0]).shape)
print(tf.gather(data, axis=1, indices=[2, 3, 7, 9, 16]).shape)
print(tf.gather(data, axis=2, indices=[2, 3, 7, 4]).shape)

# tf.grather_nd：代表从每个维度上挑选特定的index，而不是一个维度
print('-' * 20)
print(tf.gather_nd(data, indices=[0, 1]).shape)
print(tf.gather_nd(data, [0, 1, 2]).shape)
print(tf.gather_nd(data, [[0, 1, 2]]).shape)
print(tf.gather_nd(data, [[0, 0], [1, 1]]).shape)
print(tf.gather_nd(data, [[0, 0], [1, 1], [2, 2]]).shape)
print(tf.gather_nd(data, [[0, 0, 0], [1, 1, 1], [2, 2, 2]]).shape)
# 将每一个最内层看做一个整体坐标来看，比如a[0,0,0]
print(tf.gather_nd(data, [[[0, 0, 0], [1, 1, 1], [2, 2, 2]]]).shape)

# tf.boolean_mask
print('-' * 20)
data = tf.random.normal([4, 28, 28, 3])
print(tf.boolean_mask(data, mask=[True, True, False, False]).shape)
print(tf.boolean_mask(data, mask=[True, True, False], axis=3).shape)
a = tf.ones([2, 3, 4])
print(tf.boolean_mask(a, mask=[[True, False, False], [False, True, True]]).shape)  # 对应2行3列，有三个位置为真，第三维度是4，即结果为(3,4)
