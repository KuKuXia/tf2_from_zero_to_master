"""
使用tensorflow的dataset类进行数据集的预处理

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

X = tf.constant([2013, 2014, 2015, 2016, 2017])
Y = tf.constant([120, 130, 140, 150, 180])

# 当提供多个张量作为输入时，张量的第 0 维大小必须相同，且必须将多个张量作为元组（Tuple，即使用 Python 中的小括号）拼接并作为输入。
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
for x, y in dataset:
    print(x.numpy(), y.numpy())

# 加载mnist数据集
(train_x, train_y), (_, _) = tf.keras.datasets.mnist.load_data()
print(train_x.shape, train_y.shape)
train_x = tf.expand_dims(train_x.astype(np.float32) / 255., axis=-1)
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))


def show_image(dataset, batch_size, num=1):
    for image, label in dataset.take(num):
        fig, axis = plt.subplots(1, batch_size)
        for i in range(batch_size):
            axis[i].set_title(label.numpy()[i])
            axis[i].imshow(image.numpy()[i, :, :, 0])
        plt.show()
    plt.close()


# 数据集数据预处理
def rot90(image, label):
    image = tf.image.rot90(image)
    return image, label


# 使用prefetch方法预加载数据，使得GPU在训练的同时，CPU在准备数据，参数buffer_size可以手动设置，也可以设置为由TF自动选择合适的数值
# 使用shuffle操作打乱数据集
# 使用Map操作将所有图片旋转90度，其中num_parallel_calls代表多进程并行处理，可以手动设置，也可以设置为`tf.data.experimental.AUTOTUNE`让TF自动选择和合适的参数。
# 使用batch操作代表将多少个样本合并为一个元素
batch_size = 5
mnist_dataset = mnist_dataset.map(rot90, num_parallel_calls=10).shuffle(buffer_size=1000).batch(batch_size).prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)
show_image(mnist_dataset, batch_size, num=2)
