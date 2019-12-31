"""
对张量数据进行排序
"""
import os

import tensorflow as tf

tf.random.set_seed(1234)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def sort_and_argsort(a):
    # 直接对数据排序，返回排序后的数据
    print(tf.sort(a, direction='DESCENDING'))
    # 对数据排序，但是返回的是原始数据的index
    print(tf.argsort(a, direction='DESCENDING'))
    # 通过gather可以获得排序后的数据
    idx = tf.argsort(a, direction='DESCENDING')
    print(tf.gather(a, idx))

    print('-' * 10)
    a = tf.random.uniform([3, 3], maxval=10, dtype=tf.int32)
    print(a)
    print(tf.sort(a))
    print(tf.sort(a, direction='DESCENDING'))
    idx = tf.argsort(a)
    print(idx)


def top_k(a):
    # Only return top-k values and indices
    print('-' * 10)
    a = tf.random.uniform([3, 3], maxval=10, dtype=tf.int32)
    res = tf.math.top_k(a, 2)
    print(a)
    print(res.values)
    print(res.indices)

    # Top-k accuracy
    print('-' * 10)
    prob = tf.constant([[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]])
    target = tf.constant([2, 0])
    k_b = tf.math.top_k(prob, 3).indices
    print(k_b)
    k_b = tf.transpose(k_b, [1, 0])
    print(k_b)
    target_1 = tf.broadcast_to(target, [3, 2])
    print(target_1)


def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, k=max_k).indices
    pred = tf.transpose(pred, perm=[1, 0])
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)  # [k,b]

    result = []
    for k in top_k:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k * (100.0 / batch_size))
        result.append(acc)

    return result


def accuracy_demo():
    output = tf.random.normal([10, 6])
    output = tf.math.softmax(output, axis=1)
    target = tf.random.uniform([10], maxval=6, dtype=tf.int32)
    print('prob:', output.numpy())
    pred = tf.argmax(output, axis=1)
    print('pred:', pred.numpy())
    print('label:', target.numpy())
    acc = accuracy(output, target, top_k=(1, 2, 3, 4, 5, 6))
    print('top-1-6 acc:', acc)
    return acc


if __name__ == '__main__':
    a = tf.random.shuffle(tf.range(5))
    sort_and_argsort(a)
    top_k(a)
    accuracy_demo()
