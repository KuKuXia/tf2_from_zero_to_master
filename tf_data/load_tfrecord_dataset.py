import os

import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

data_dir = '../data/cats-vs-dogs'
tf_record_file = data_dir + '/train/train.tfrecords'

raw_dataset = tf.data.TFRecordDataset(tf_record_file)
feature_description = {  # 定义Feature结构，告诉编码器每隔Feature的类型是什么
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}


def _parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])  # 解码jpeg图片
    return feature_dict['image'], feature_dict['label']


dataset = raw_dataset.map(_parse_example).shuffle(buffer_size=10000)

for image, label in dataset:
    plt.title('cat' if label == 0 else 'dog')
    plt.imshow(image.numpy())
    plt.show()
