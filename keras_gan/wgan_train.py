"""
WGAN和GAN的神经网络结构是一样的，区别在于loss
"""

import glob
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from keras_gan.dataset import make_anime_dataset
from keras_gan.gan import Generator, Discriminator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpu)
tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

tf.random.set_seed(22)
np.random.seed(22)


def save_result(val_out, val_block_size, image_path, color_mode):
    """
    将batch图片合并成一张图片并且保存
    :param val_out:
    :param val_block_size:
    :param image_path:
    :param color_mode:
    :return:
    """

    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)


def celoss_ones(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def gradient_penalty(discriminator, batch_x, fake_image):
    batchsz = batch_x.shape[0]

    # [b, h, w, c]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # [b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate, training=True)
    grads = tape.gradient(d_interplote_logits, interplate)

    # grads:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    gp = tf.reduce_mean((gp - 1) ** 2)

    return gp


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 1. treat real image as real
    # 2. treat generated image as fake
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)  # 正样本判断为真
    d_loss_fake = celoss_zeros(d_fake_logits)  # 负样本判断为假

    gp = gradient_penalty(discriminator, batch_x, fake_image)

    loss = d_loss_fake + d_loss_real + 10. * gp  # 判别器的目标是将真的判断为真，假的判断为假，WGAN多了Gradient Penalty

    return loss, gp


def g_loss_fn(generator, discriminator, batch_z, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)  # 这个地方的目标是欺骗生成器，使得生成的图片是真的，因此label为1

    return loss


def main():
    # hyper parameters
    z_dim = 100
    epochs = 3000000
    batch_size = 512
    learning_rate = 0.0001
    is_training = True

    img_path = glob.glob(
        r'/media/Data_1/LongXiaJun/faces/*.jpg')

    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)
    print(dataset, img_shape)
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(),
          tf.reduce_min(sample).numpy())
    dataset = dataset.repeat()  # 无限制迭代数据集
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):

        # 随机采样z向量作为generator的输入
        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        batch_x = next(db_iter)

        # train D
        with tf.GradientTape() as tape:
            d_loss, gp = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss), 'gp: ', float(gp))

            z = tf.random.uniform([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = '../logs/wgan_images/wgan-%d.png' % epoch
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')


if __name__ == '__main__':
    main()
