import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)


@tf.function
def square_if_positive(x):
    if x > 0:
        x = x * x
    else:
        x = 0
    return x


a = tf.constant(1)
b = tf.constant(-1)
print(square_if_positive(a), square_if_positive(b))
print(tf.autograph.to_code(square_if_positive.python_function))


# TensorArray
@tf.function
def array_write_and_read():
    arr = tf.TensorArray(dtype=tf.float32, size=3)
    arr = arr.write(0, tf.constant(0.0))
    arr = arr.write(1, tf.constant(1.0))
    arr = arr.write(2, tf.constant(2.0))
    arr_0 = arr.read(0)
    arr_1 = arr.read(1)
    arr_2 = arr.read(2)
    return arr_0, arr_1, arr_2


a, b, c = array_write_and_read()
print(a, b, c)
