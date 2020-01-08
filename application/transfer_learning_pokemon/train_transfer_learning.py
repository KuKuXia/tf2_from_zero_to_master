import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

print(f'GPU: {gpus}')

from application.transfer_learning_pokemon.pokemon import load_pokemon, normalize
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(22)
np.random.seed(22)


def preprocess(x, y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)  # RGBA
    x = tf.image.resize(x, [244, 244])

    # x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [224, 224, 3])

    # x: [0,255]=> -1~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=5)

    return x, y


batch_size = 128

# 创建训练集Dataset对象
images, labels, table = load_pokemon('/media/Data_1/LongXiaJun/pokemon', mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batch_size)

# 创建验证集Dataset对象
images2, labels2, table = load_pokemon('/media/Data_1/LongXiaJun/pokemon', mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batch_size)

# 创建测试集Dataset对象
images3, labels3, table = load_pokemon('/media/Data_1/LongXiaJun/pokemon', mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batch_size)

#
net = keras.applications.VGG19(weights='imagenet', include_top=False,
                               pooling='max')
net.trainable = False
newnet = keras.Sequential([
    net,
    layers.Dense(5)
])
newnet.build(input_shape=(4, 224, 224, 3))
newnet.summary()

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.01,
    patience=10
)

newnet.compile(optimizer=optimizers.Adam(lr=1e-3),
               loss=losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
newnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=100,
           callbacks=[early_stopping])
newnet.evaluate(db_test)
