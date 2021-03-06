import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'

num_epochs = 5
batch_size_per_replica = 256
learning_rate = 0.001

num_workers = 2

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:20000", "180.201.13.2:20001"]
    },
    'task': {'type': 'worker', 'index': 0}
})

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
batch_size = batch_size_per_replica * num_workers


def resize(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.0
    return image, label


dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)

# 切记这个地方要添加repeat(),不然只能训练一个epoch,然后会报错
dataset = dataset.map(resize).shuffle(1024).batch(batch_size).repeat()

with strategy.scope():
    model = tf.keras.applications.MobileNetV2()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

model.fit(dataset, epochs=num_epochs)
