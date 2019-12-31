"""
Tensorflow数学运算
"""
import tensorflow as tf

a = tf.ones([2, 2])
b = tf.fill([2, 2], 2.)

# +-*/ // %
print(f'a: {a}\nb: {b}')
print(f'a + b: {a + b}\na - b: {a - b}\na/b: {a / b}')
print(f'b//a: {b // a}\nb%a: {b % a}')

# tf.math.log, tf.exp
print(tf.math.log(a))
print(tf.exp(a))

# log2, log10   log_10(b)/log_10(c) = log_c(b)
print(tf.math.log(8.0) / tf.math.log(2.))
print(tf.math.log(100.0) / tf.math.log(10.))

# pow, sqrt
print(tf.pow(b, 3))
print(b ** 3)
print(tf.sqrt(b))

# @ matmul
print(a @ b)
print(tf.matmul(a, b))

a = tf.ones([4, 2, 3])
b = tf.fill([4, 3, 5], 2.)
print(f'a: {a}\n b:{b}')
print(a @ b)
print(tf.matmul(a, b))
print(tf.matmul(a, b).shape)

c = tf.fill([3, 5], 2.)
print(f'shape of c: {c.shape}')
c_1 = tf.broadcast_to(b, [4, 3, 5])
print(a @ c)
print(a @ c_1)

# @矩阵运算
# Y = X@W + b
x = tf.ones([4, 2])
W = tf.ones([2, 1])
b = tf.constant(0.1)
print(f'x: {x}, W: {W}, b: {b}')
print(x @ W + b)

out = x @ W + b
y = tf.nn.relu(out)
print(y)
