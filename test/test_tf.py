import sys
from os.path import abspath, join, dirname

sys.path.insert(0, join(abspath(dirname(__file__)), '../'))

from tf_basics.tf_01_linear_regression import run


def test_run():
    assert run() == (0.089, 1.478)
