import numpy as np
import tensorflow as tf

# Tensor Property
print("Tensor Property: ")
with tf.device('cpu'):
    a = tf.constant([1])

with tf.device('gpu'):
    b = tf.range(4)

print(a.device)  # 查看数据存储在cpu还是gpu上面
print(b.device)

aa = a.gpu()  # 将cpu数据转为gpu数据
bb = b.cpu()
print(aa.device)
print(bb.device)

print(b.numpy())  # 将tensor转换为numpy
print(b.ndim)  # 查看数据维度
print(tf.rank(b))  # 查看数据秩
# print(b.name)  # 在eager execution模式下没有意义，就是b 会报错

# Check Tensor Type
print('Check tensor type: ')
a = tf.constant([1.])
b = tf.constant([True, False])
c = tf.constant('hello world.')
d = np.arange(4)

print(isinstance(a, tf.Tensor))
print(tf.is_tensor(b))
print(tf.is_tensor(d))

print(f'{a.dtype, b.dtype, c.dtype}')
print(a.dtype == tf.float32)
print(c.dtype == tf.string)

# Convert
# int <-> float
print('Convert int to float: ')
a = np.arange(5)
print(a.dtype)

aa = tf.convert_to_tensor(a)
print(aa)

aa_2 = tf.convert_to_tensor(a, dtype=tf.int32)
print(aa_2)

aa_3 = tf.cast(aa, dtype=tf.float32)
print(aa_3)

aa_4 = tf.cast(aa, dtype=tf.float64)
print(aa_4)

aa_5 = tf.cast(aa_4, dtype=tf.int32)
print(aa_5)

# int <-> bool
print("Convert int to float: ")
b = tf.constant([0, 1])
bb = tf.cast(b, dtype=tf.bool)
print(bb)

b_1 = tf.cast(bb, dtype=tf.int32)
print(b_1)

# tf.Variable
print('TF variable')
a = tf.range(5)
print(a)

b = tf.Variable(a)
print(b.dtype)
print(b.name)

b = tf.Variable(a, name='input_data')
print(b)
print(b.name)
print(b.trainable)

print(isinstance(b, tf.Tensor))  # 返回为False，出错了，不推荐使用
print(isinstance(b, tf.Variable))
print(tf.is_tensor(b))

# To numpy

print('To numpy: ')
print(b.numpy())
