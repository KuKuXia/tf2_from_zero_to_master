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
