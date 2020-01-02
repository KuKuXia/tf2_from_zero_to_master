import tensorflow as tf
from tf_01_linear_regression import run

tf.random.set_seed(1234)


def test_run():
    assert run() == (0.089, 1.478)
