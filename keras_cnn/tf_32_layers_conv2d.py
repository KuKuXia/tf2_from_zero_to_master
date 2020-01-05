import os

import tensorflow as tf
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.random.normal([1, 28, 28, 3])
layer = layers.Conv2D(4, kernel_size=5, strides=1, padding='valid')
out = layer(x)
print(out)
