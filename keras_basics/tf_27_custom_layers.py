import os

import tensorflow as tf
from tensorflow.keras import datasets, layers, Sequential, Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    """
    :param x: 
    :param y: 
    :return: 
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batch_size = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('Datasets: ', x.shape, y.shape, x.min(), y.max())

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batch_size)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batch_size)

sample = next(iter(db))
print(sample[0].shape, sample[1].shape)

network = Sequential(layers=[
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])

network.build(input_shape=(None, 28 * 28))
network.summary()


# 自定义神经网络结构
class MyDense(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w', [input_dim, output_dim])
        self.bias = self.add_weight('b', [output_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.fc1 = MyDense(28 * 28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x


print('Custom Networks')
network_1 = MyModel()
network_1.compile(optimizer=tf.optimizers.Adam(lr=0.01), loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
network_1.fit(db, epochs=5, validation_data=ds_val, validation_freq=2)
network_1.evaluate(ds_val)
network_1.summary()

sample = next(iter(ds_val))
x = sample[0]
y = sample[1]
pred = network.predict(x)
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)
print(pred)
print(y)
