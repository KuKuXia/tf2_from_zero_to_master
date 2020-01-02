import os

import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, optimizers


def preprocess(x, y):
    """
    :param x: is a simple image, not a batch
    :param y: is a label
    :return: x,y
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('Datasets: ', x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batch_size)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batch_size)

sample = next(iter(db))
print(sample[0].shape, sample[1].shape)

network = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])

network.build(input_shape=(None, 28 * 28))
network.summary()

network.compile(optimizers=optimizers.Adam(lr=0.01), loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(db, epochs=5, validation_data=ds_val, validation_freq=2)

network.evaluate(ds_val)

sample = next(iter(ds_val))  # sample: (128, 784) (128, 10)

x = sample[0]
y = sample[1]
print(x.shape)
print(y.shape)
pred = network.predict(x)  # [b, 10]
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)
print(pred)
print(y)
