import os

import tensorflow as tf
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

num_epochs = 10
batch_size = 32
learning_rate = 0.001


def _decode_and_resize(image, label):
    image_resized = tf.image.resize(image, [256, 256]) / 255.0
    return image_resized, label


if __name__ == '__main__':
    dataset_train = tfds.load('cats_vs_dogs', split=tfds.Split.TRAIN, as_supervised=True)
    dataset_train = dataset_train.map(map_func=_decode_and_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_train = dataset_train.shuffle(buffer_size=23000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.build(input_shape=(None, 256, 256, 3))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])

    model.fit(dataset_train, epochs=num_epochs)
    print(model.metrics_names)
    print(model.evaluate(dataset_train))
