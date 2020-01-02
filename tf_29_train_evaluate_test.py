import os

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batch_size = 128
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())

# 将训练集随机拆分为两个部分，train and val
idx = tf.range(60000)
idx = tf.random.shuffle(idx)
x_train, y_train = tf.gather(x, idx[:50000]), tf.gather(y, idx[:50000])
x_val, y_val = tf.gather(x, idx[-10000:]), tf.gather(y, idx[-10000:])
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.map(preprocess).shuffle(50000).batch(batch_size)

db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).shuffle(10000).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batch_size)

sample = next(iter(db_train))
print(sample[0].shape, sample[1].shape)

# 添加L2正则化和dropout防止过拟合
network = Sequential([layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                      layers.Dropout(0.3),  # 0.5 rate to drop
                      layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                      layers.Dropout(0.2),  # 0.5 rate to drop
                      layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                      layers.Dropout(0.2),  # 0.5 rate to drop
                      layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                      layers.Dense(10)])
network.build(input_shape=(None, 28 * 28))
network.summary()

network.compile(optimizer=optimizers.Adam(lr=0.003),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

network.fit(db_train, epochs=50, validation_data=db_val, validation_freq=2)

print('Test performance:')
network.evaluate(db_test)

sample = next(iter(db_test))
x = sample[0]
y = sample[1]  # one-hot
pred = network.predict(x)  # [b, 10]
# convert back to number
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)

print(pred)
print(y)
