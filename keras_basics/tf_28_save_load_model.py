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


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)

sample = next(iter(db))
print(sample[0].shape, sample[1].shape)

network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])
network.build(input_shape=(None, 28 * 28))
network.summary()

network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

network.fit(db, epochs=3, validation_data=ds_val, validation_freq=2)

network.evaluate(ds_val)

# 保存整个网络结构和权重，不需要重新定义build模型结构
network.save('../model/model.h5')
print('saved total model.')
del network

print('loaded model from file.')
network = tf.keras.models.load_model('../model/model.h5', compile=False)
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

network.evaluate(ds_val)

# 生产环境部署
print('Save served model')
tf.saved_model.save(network, '../model/server_model')
print('Delete model')
del network

print('Load served model')
imported = tf.saved_model.load('../model/server_model')
f = imported.signatures['serving_default']

# 直接inference输出预测结果
# 随机生成一个全是1的张量
pred = f(input_1=tf.ones([1, 784]))
print(tf.argmax(pred['output_1'], axis=1))
# 抽取ds_val中第一张图片，扩展为一个张量
pred = f(input_1=tf.expand_dims(sample[0][0], axis=0))  # 需要扩展为[1, 784]
print(tf.argmax(pred['output_1'], axis=1))
# 实际label
print(tf.argmax([sample[1][0]], axis=1))
