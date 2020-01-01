"""
数据集加载
"""
import os

import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# mnist dataset
(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f'shape of x: {x.shape}, y: {y.shape}')
print(f'shape of x_test: {x_test.shape}, y_test: {y_test.shape}')
print(y[:4])
y_onehot = tf.one_hot(y, depth=10)
print(y_onehot[:2])

# cifar10/100 dataset
(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(f'shape of x: {x.shape}, y: {y.shape}')
print(f'shape of x_test: {x_test.shape}, y_test: {y_test.shape}')
print(y[:4])

# tf.data.Dataset
# from_tensor_slices()
db = tf.data.Dataset.from_tensor_slices(x_test)
print(next(iter(db)).shape)
# can not use [x_test, y_test]
db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
print(next(iter(db))[0].shape)

# .shuffle 随机打散数据
db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db = db.shuffle(10000)


# .map
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


db2 = db.map(preprocess)
res = next(iter(db2))
print(res[0].shape, res[1].shape)
print(res[1][0][:2])

# .batch
db3 = db2.batch(32)
res = next(iter(db3))
print(f'shape of res[0]: {res[0].shape}, res[1]: {res[1].shape}')

# StopIteration
# db_iter = iter(db3)
# while True:
#     next(db_iter) # 会报错

# .repeat
db4 = db3.repeat()  # 永远不会退出
db4 = db3.repeat(2)  # 重复整个数据集2次


# Simple example

def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.
    y = tf.cast(y, tf.int64)
    return x, y


def mnist_dataset():
    (x, y), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
    y = tf.one_hot(y, depth=10)
    y_val = tf.ones(y_val, depth=10)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.shuffle(60000).batch(100)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(prepare_mnist_features_and_labels)
    ds_val = ds_val.shuffle(10000).batch(100)
    return ds, ds_val
