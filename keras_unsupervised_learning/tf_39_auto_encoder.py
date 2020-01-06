import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import Sequential, layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(22)
np.random.seed(22)


def save_images(imgs, name):
    new_img = Image.new('L', (280, 280))
    index = 0

    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_img.paste(im, (i, j))
            index += 1
    new_img.save(name)


hidden_dim = 20
batch_size = 512
lr = 1e-2

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 225.

# we do not need label
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batch_size * 5).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batch_size)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


class AE(keras.Model):
    def __init__(self):
        super(AE, self).__init__()

        # Encoders
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(hidden_dim)

        ])

        # Decoders
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None):
        # [b, 784] => [b,10]
        h = self.encoder(inputs)
        # [b, 10] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat


def main():
    model = AE()
    model.build(input_shape=(None, 784))
    model.summary()

    optimizer = tf.optimizers.Adam(lr=lr)

    for epoch in range(100):
        for step, x in enumerate(train_db):
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, [-1, 784])
            with tf.GradientTape() as tape:
                x_rec_logit = model(x)
                # 可以为MSE等Loss
                rec_loss = tf.losses.binary_crossentropy(x, x_rec_logit, from_logits=True)
                rec_loss = tf.reduce_mean(rec_loss)
            grads = tape.gradient(rec_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(f'epoch: {epoch}, step: {step}, loss: {float(rec_loss)}')

        if epoch % 10 == 0:
            # evaluation
            x = next(iter(test_db))
            logits = model(tf.reshape(x, [-1, 784]))
            x_hat = tf.sigmoid(logits)
            # [b, 784] => [b, 28,28]
            x_hat = tf.reshape(x_hat, [-1, 28, 28])

            # [b, 28, 28] => [2b, 28,28]
            x_concat = tf.concat([x, x_hat], axis=0)
            # x_concat = x_hat
            print(x_concat.shape)
            x_concat = x_concat.numpy() * 255.
            x_concat = x_concat.astype(np.uint8)
            save_images(x_concat, './../logs/ae_images/rec_epoch_%d.png' % epoch)


if __name__ == '__main__':
    main()
