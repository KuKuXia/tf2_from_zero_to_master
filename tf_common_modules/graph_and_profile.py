import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)


# 数据获取以及预处理
class MNISTLoader():
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


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        # Flatten层将除第一维外的维度展平
        self.flatten = tf.keras.layers.Flatten()  # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):  # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)  # [batch_size, 784]
        x = self.dense1(x)  # [batch_size, 100]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


num_batches = 1000
batch_size = 50
learning_rate = 0.001
log_dir = './../logs/graph'
model = MLP()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
summary_writer = tf.summary.create_file_writer(log_dir)  # 实例化记录器
tf.summary.trace_on(graph=True, profiler=True)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred))
        print(f'batch: {batch_index}, loss: {loss.numpy()}')
        with summary_writer.as_default():  # 制定记录器
            tf.summary.scalar('loss', loss, step=batch_index)  # 将当前损失函数的值写入记录器
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

with summary_writer.as_default():
    tf.summary.trace_export(name='model_trace', step=0, profiler_outdir=log_dir)  # 保存trace记录
