import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 列出当前主机设备列表
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus)
print(cpus)

# 设置当前脚本可见的设备
# tf.config.experimental.set_visible_devices(device_type='GPU', devices=gpus[0:2])

# 可以通过环境变量的方式设置可见的设备
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# 设置显存使用策略
# 方式1：动态申请显存
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

# 方式2：固定显存大小，超出将会报错
# 可以理解为建立了一个显存大小为1GB的虚拟GPU
for gpu in gpus:
    tf.config.experimental.set_virtual_device_configuration(gpu, [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# 单GPU模拟多GPU环境
# 在实体GPU2号上面建立4个显存均为2GB的虚拟GPU
tf.config.experimental.set_virtual_device_configuration(gpus[2], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

logical_gpus = tf.config.experimental.list_logical_devices(device_type='GPU')
print(len(logical_gpus))

# 常用方法
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
