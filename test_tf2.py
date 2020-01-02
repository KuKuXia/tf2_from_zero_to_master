import tensorflow as tf
from tf_01_linear_regression import run
from tf_12_sort_and_topk import accuracy_demo

tf.random.set_seed(1234)


def test_run():
    assert run() == (0.089, 1.478)


def test_accuracy():
    assert accuracy_demo() == [10.0, 20.0, 20.0, 50.0, 70.0, 100.0]
