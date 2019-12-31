"""
Norm：矩阵范数
"""
import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# L2 Norm
print('-' * 20, '\nL2 norm')
a = tf.ones([2, 2])
print(tf.norm(a))
print(tf.norm(a, ord=2, axis=1))
print(tf.sqrt(tf.reduce_sum(tf.square(a))))

b = tf.ones([4, 28, 28, 3])
print(tf.norm(b))
print(tf.sqrt(tf.reduce_sum(tf.square(b))))

# L1 Norm
print('-' * 20, '\n L1 norm')
print(tf.norm(a, ord=1))
print(tf.norm(a, ord=1, axis=0))
print(tf.norm(a, ord=1, axis=1))

# reduce_min/max/mean: reduce的意思是提醒会产生降维操作
print('-' * 20, '\ntf.reduce_min/max/mean/sum')
a = tf.random.normal([4, 10])
print(a)
# 不指定维度axis，默认是求整个张量a中的统计数据
print(tf.reduce_min(a), tf.reduce_max(a), tf.reduce_mean(a), tf.reduce_sum(a))

# argmax/argmin
print('-' * 20, '\ntf.argmax/min')
print(tf.argmax(a))
print(tf.argmax(a).shape)
print(tf.argmin(a))
print(tf.argmin(a).shape)

# tf.equal
print('-' * 20, '\ntf.equal')
a = tf.constant([1, 2, 3, 4, 5])
b = tf.constant([1, 2, 3, 4, 5])
c = tf.range(5)
print(tf.equal(a, b))
print(tf.equal(a, c))
res_a_b = tf.equal(a, b)
res_a_c = tf.equal(a, c)
print(tf.reduce_sum(tf.cast(res_a_b, dtype=tf.int32)))
print(tf.reduce_sum(tf.cast(res_a_c, dtype=tf.int32)))

# Accuracy
print('-' * 20, '\nAccuracy')
a = tf.convert_to_tensor([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=tf.float32)
pred = tf.cast(tf.argmax(a, axis=1), dtype=tf.int32)
y = tf.convert_to_tensor([2, 1])
print(tf.equal(y, pred))
correct = tf.reduce_sum(tf.cast(tf.equal(y, pred), dtype=tf.int32))
print(correct)
print(correct / 2)

# tf.unique
print('-' * 20, '\ntf.unique')
a = tf.range(5)
print(tf.unique(a))
a = tf.constant([4, 2, 2, 4, 3])
print(tf.unique(a))
