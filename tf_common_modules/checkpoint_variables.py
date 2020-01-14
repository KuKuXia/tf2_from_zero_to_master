import argparse
import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument('--num_epochs', default=100)
parser.add_argument('--batch_size', default=500)
parser.add_argument('--learning_rate', default=0.001)
args = parser.parse_args()


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


def train():
    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    num_batches = int(data_loader.num_train_data // args.batch_size * args.num_epochs)
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)  # 实例化Checkpoint，设置保存对象为model
    manager = tf.train.CheckpointManager(checkpoint, directory='./save', checkpoint_name='model.ckpt',
                                         max_to_keep=5)  # 使用tf.train.CheckpointManager管理Checkpoint，只保存最新的k个模型参数
    for batch_index in range(1, num_batches + 1):
        X, y = data_loader.get_batch(args.batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred))
        print(f'Batch: {batch_index}, loss: {loss.numpy()}')
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if batch_index % 1000 == 0:
            # 使用CheckpointManager保存模型参数到文件并自定义编号
            path = manager.save(checkpoint_number=batch_index)
            print(f'model saved to {path}')


def model_test():
    model_to_be_resorted = MLP()
    # 实例化Checkpoint，设置恢复对象为新建立的模型model_to_be_restored
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_resorted)
    checkpoint.restore(tf.train.latest_checkpoint('./save'))  # 从文件恢复模型参数
    y_pred = np.argmax(model_to_be_resorted.predict(data_loader.test_x), axis=-1)
    print(f'test accuracy: {sum(y_pred == data_loader.test_y) / data_loader.num_test_data}')


if __name__ == '__main__':

    data_loader = MNISTLoader()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        model_test()
    else:
        print('Please choose the mode to train or test')
