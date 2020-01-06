import os

import tensorflow as tf
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.range(5)
x = tf.random.shuffle(x)
print(x)

net = layers.Embedding(10, 4)  # 10代表整个语料库有10个单词，4代表将每个单词映射到4维的向量表示
print(net(x))
print(net.trainable)
print(net.trainable_variables)
