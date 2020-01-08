import os

import numpy as np
import tensorflow as tf
from application.transfer_learning_pokemon.pokemon import load_pokemon, normalize
from application.transfer_learning_pokemon.resnet import ResNet
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(22)
np.random.seed(22)

# 设置GPU显存按需分配
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def preprocess(x, y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)  # RGBA
    # 图片缩放
    # x = tf.image.resize(x, [244, 244])
    # 图片旋转
    # x = tf.image.rot90(x,2)
    # 随机水平翻转
    x = tf.image.random_flip_left_right(x)
    # 随机竖直翻转
    # x = tf.image.random_flip_up_down(x)

    # 图片先缩放到稍大尺寸
    x = tf.image.resize(x, [244, 244])
    # 再随机裁剪到合适尺寸
    x = tf.image.random_crop(x, [224, 224, 3])

    # x: [0,255]=> -1~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=5)

    return x, y


batch_size = 256

# creat train db
images, labels, table = load_pokemon('/media/Data_1/LongXiaJun/pokemon', mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batch_size)

# crate validation db
images2, labels2, table = load_pokemon('/media/Data_1/LongXiaJun/pokemon', mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batch_size)

# create test db
images3, labels3, table = load_pokemon('/media/Data_1/LongXiaJun/pokemon', mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batch_size)

# 自定义简单神经网络结构
resnet = keras.Sequential([
    layers.Conv2D(16, 5, 3),
    layers.MaxPool2D(3, 3),
    layers.ReLU(),
    layers.Conv2D(64, 5, 3),
    layers.MaxPool2D(2, 2),
    layers.ReLU(),
    layers.Flatten(),
    layers.Dense(64),
    layers.ReLU(),
    layers.Dense(5)
])

# ResNet
resnet = ResNet(5)
resnet.build(input_shape=(4, 224, 224, 3))
resnet.summary()

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=5
)

resnet.compile(optimizer=optimizers.Adam(lr=1e-3),
               loss=losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
resnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=100,
           callbacks=[early_stopping])
resnet.evaluate(db_test)
