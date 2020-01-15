import os
import time

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)


# 数据获取以及预处理
class MNISTLoader:
    def __init__(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = tf.keras.datasets.mnist.load_data()
        # MNIST中的图像默认为uint8 (0-255)的数字，需要归一化到0-1之间的浮点数，并且在最后添加一层颜色通道
        self.train_x = np.expand_dims(self.train_x.astype(np.float32) / 255., -1)
        self.test_x = np.expand_dims(self.test_x.astype(np.float32) / 255., -1)
        self.train_y = self.train_y.astype(np.int32)
        self.test_y = self.test_y.astype(np.int32)
        self.num_train_data, self.num_test_data = self.train_x.shape[0], self.test_x.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_x)[0], batch_size)
        return self.train_x[index, :], self.train_y[index]


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32,  # 卷积层神经元（卷积核）数目
                                            kernel_size=[5, 5],  # 感受野大小
                                            padding='same',  # padding策略（vaild 或 same）
                                            activation=tf.nn.relu  # 激活函数
                                            )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(64, 5, padding='same', activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)  # [batch_size, 28, 28, 32]
        x = self.pool1(x)  # [batch_size, 14, 14, 32]
        x = self.conv2(x)  # [batch_size, 14, 14, 64]
        x = self.pool2(x)  # [batch_size, 7, 7, 64]
        x = self.flatten(x)  # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)  # [batch_size, 1024]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


num_batches = 4000
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()


@tf.function
def train_one_step(X, y):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        # 注意这里使用了TensorFlow内置的tf.print()。@tf.function不支持Python内置的print方法
        tf.print("loss", loss)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


if __name__ == '__main__':
    model = CNN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    start_time = time.time()
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        train_one_step(X, y)
    end_time = time.time()
    print(end_time - start_time)
