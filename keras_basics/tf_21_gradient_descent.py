"""
梯度下降
"""

import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w = tf.constant(1.)
x = tf.constant(2.)
b = tf.constant(6.)
y = x * w

with tf.GradientTape() as tape:
    tape.watch([w])
    y2 = x * w
grad1 = tape.gradient(y, [w])
print(grad1)

with tf.GradientTape() as tape:
    tape.watch([w])
    y2 = x * w

grad2 = tape.gradient(y2, [w])
print(grad2)

# 二阶求导，基本上用不到
with tf.GradientTape() as t1:
    t1.watch([w, b])  # 很重要，跟踪梯度信息
    with tf.GradientTape() as t2:
        t2.watch([w, b])  # 很重要，跟踪梯度信息
        y4 = x * w ** 2 + 2 * b
        dy_dw, dy_db = t2.gradient(y4, [w, b])
d2y_dw2 = t1.gradient(dy_dw, w)
print(dy_dw, dy_db)
print(d2y_dw2)


# Persistent GradientTape
# grad2 = tape.gradient(y2, [w]) # 报错 RuntimeError: GradientTape.gradient can only be called once on non-persistent tapes.
# 设置persistent=True，后面可以多次调用
def show_gradient_persistent():
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([w])
        y3 = x * w

    grad3 = tape.gradient(y3, [w])
    print(grad3)

    grad3 = tape.gradient(y3, [w])
    print(grad3)


# f(x) = tf.sigmoid
# f'(x) = f(x) * (1 - f(x))
print('-' * 50)
x = tf.linspace(-10., 10., 10)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.sigmoid(x)
grads = tape.gradient(y, [x])
grads_1 = tf.sigmoid(x) * (1 - tf.sigmoid(x))
print('x: ', x)
print('y: ', y)
print('grads: ', grads)
print('grads_1: ', grads_1)
print(np.all(grads_1 == grads))  # 精度问题，实际上相等的

# f(x) = tf.tanh
# f'(x) = 1 - f(x)^2
print('-' * 50)
x = tf.linspace(-5., 5., 10)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.tanh(x)
grads = tape.gradient(y, [x])
grads_1 = 1 - tf.tanh(x) ** 2
print('x: ', x)
print('y: ', y)
print('grads: ', grads)
print('grads_1: ', grads_1)
print(np.all(grads_1 == grads))

# f(x) = tf.nn.relu
# f'(x) = 1 - f(x)^2
print('-' * 50)
x = tf.linspace(-5., 5., 10)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.nn.relu(x)
grads = tape.gradient(y, [x])
mask = tf.nn.relu(x) > 0
x_1 = tf.ones(x.shape)
x_2 = tf.zeros(x.shape)
grads_1 = tf.where(mask, x_1, x_2)
print('x: ', x)
print('y: ', y)
print('grads: ', grads)
print('grads_1: ', grads_1)
print(np.all(grads_1 == grads))

print('tf.leaky_relu: ', tf.nn.leaky_relu(x, alpha=0.4))

x = tf.random.normal([2, 4])
w = tf.random.normal([4, 3])
b = tf.zeros([3])
y = tf.constant([2, 0])


# MSE gradient
def mse_gradient():
    print('-' * 50)
    print('MSE Gradient')

    with tf.GradientTape() as tape:
        tape.watch([w, b])
        prob = tf.nn.softmax(x @ w + b, axis=1)
        loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y, depth=3), prob))
    grads = tape.gradient(loss, [w, b])
    print(grads[0])
    print(grads[1])


# Crossentropy gradient
def crossentropy_gradient():
    print('-' * 50)
    print('Crossentropy Gradient')

    with tf.GradientTape() as tape:
        tape.watch([w, b])
        logits = x @ w + b
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(tf.one_hot(y, depth=3), logits, from_logits=True))
    grads = tape.gradient(loss, [w, b])
    print(grads[0])
    print(grads[1])


# single output perceptron gradient
def single_layer_perceptron_gradient():
    print('-' * 50)
    print('Single output perceptron Gradient')

    x = tf.random.normal([1, 3])
    w = tf.random.normal([3, 1])
    b = tf.zeros([1])
    y = tf.constant([1])

    with tf.GradientTape() as tape:
        tape.watch([w, b])
        logits = tf.sigmoid(x @ w + b)
        loss = tf.reduce_mean(tf.losses.MSE(y, logits))
    grads = tape.gradient(loss, [w, b])
    print(grads[0])
    print(grads[1])


# chain rule
def chain_rule():
    x = tf.constant(1.)
    w1 = tf.constant(2.)
    b1 = tf.constant(1.)
    w2 = tf.constant(2.)
    b2 = tf.constant(1.)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([w1, b1, w2, b2])

        y1 = x * w1 + b1
        y2 = y1 * w2 + b2
    dy2_dy1 = tape.gradient(y2, [y1])[0]
    dy1_dw1 = tape.gradient(y1, [w1])[0]
    dy2_dw1 = tape.gradient(y2, [w1])[0]
    print(dy2_dy1, dy1_dw1, dy2_dw1)


mse_gradient()
crossentropy_gradient()
single_layer_perceptron_gradient()
chain_rule()
