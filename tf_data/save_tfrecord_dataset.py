"""
Download the dataset from:
https://www.floydhub.com/fastai/datasets/cats-vs-dogs
"""

import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

data_dir = '../data/cats-vs-dogs'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
tf_record_file = data_dir + '/train/train.tfrecords'

train_cat_filenames = [train_cats_dir + filename for filename in os.listdir(train_cats_dir)]
train_dog_filenames = [train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)]
train_filenames = train_cat_filenames + train_dog_filenames
train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)

with tf.io.TFRecordWriter(tf_record_file) as writer:
    for filename, label in zip(train_filenames, train_labels):
        image = open(filename, 'rb').read()  # 读取数据集图片到内存，image为一个Byte类型的字符串
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # 标签是一个 Int 对象
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立Example对象
        writer.write(example.SerializeToString())  # 将Example对象序列化并且写入TFRecord文件
