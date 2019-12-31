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

# log2, log10
print(tf.math.log(8.0) / tf.math.log(2.))
print(tf.math.log(100.0) / tf.math.log(10.))

# pow, sqrt
