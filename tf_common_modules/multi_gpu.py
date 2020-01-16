import os

import tensorflow as tf
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

num_epochs = 5
batch_size_per_replica = 64
learning_rate = 1e-3

strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:2'])
print(f'Number of devices: {strategy.num_replicas_in_sync}')
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync


# 载入数据集并且预处理
def resize(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.
    return image, label


# 当as_supervised为True时，返回image和label两个键值
dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(resize).shuffle(1024).batch(batch_size)

with strategy.scope():
    model = tf.keras.applications.MobileNetV2()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

model.fit(dataset, epochs=2)
model.evaluate(dataset)
