"""
损失函数
"""

import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

y = tf.constant([1, 2, 3, 0, 2])
y = tf.one_hot(y, depth=4)
y = tf.cast(y, dtype=tf.float32)
out = tf.random.normal([5, 4])
loss1 = tf.reduce_mean(tf.square(y - out))
loss2 = tf.square(tf.norm(y - out)) / (5 * 4)
loss3 = tf.reduce_mean(tf.losses.MSE(y, out))
print(f'loss1: {loss1}, loss2: {loss2}, loss3: {loss3}')  # MSE is a function, MeanSquareError is a class

# Entropy，注意是以2为底计算，但是tf是使用e为底计算的，所以需要除以loge(2)
# -sum(p(i)*log p(i))

# 熵大，分布均匀
a = tf.fill([4], 0.25)
print(a * tf.math.log(a) / tf.math.log(2.))
print(-tf.reduce_sum(a * tf.math.log(a) / tf.math.log(2.)))

a = tf.constant([0.1, 0.1, 0.1, 0.7])
print(-tf.reduce_sum(a * tf.math.log(a) / tf.math.log(2.)))

# 熵小，不稳定
a = tf.constant([0.01, 0.01, 0.01, 0.97])
print(-tf.reduce_sum(a * tf.math.log(a) / tf.math.log(2.)))

# Categorical Cross Entropy
print('Cross Entropy')
# 小写字母的是函数，可以直接调用
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.25, 0.25, 0.25, 0.25]))
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.1, 0.1, 0.8, 0]))
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.1, 0.1, 0.1, 0.7]))
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.1, 0.7, 0.1, 0.1]))

# 带有大写字母的是类名称，需要先实例化
binary_cross_entropy = tf.losses.BinaryCrossentropy()
print(binary_cross_entropy([1], [0.1]))
print(binary_cross_entropy([1], [0.8]))

# 小写字母的是函数，可以直接调用
print(tf.losses.binary_crossentropy([1], [0.1]))
print(tf.losses.binary_crossentropy([1], [0.8]))

# 因为在计算交叉熵的时候，自己写的代码从进行softmax操作出现除以0的不稳定情况，因此最好不要做softmax操作，而是选择直接使用tf提供的封装函数。
print('-' * 50)
print('Numerical Stability')
x = tf.random.normal([1, 784])
w = tf.random.normal([784, 2])
b = tf.zeros([2])

logits = x @ w + b
print(logits)

prob = tf.math.softmax(logits, axis=1)
print(prob)

print(tf.losses.categorical_crossentropy([0, 1], logits, from_logits=True))
print(tf.losses.categorical_crossentropy([0, 1], prob))
