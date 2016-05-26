import numpy as np


def test_mnist():
    from nolearn_utils.datasets import MNIST

    dataset = MNIST()
    assert dataset.X.shape == (60000, 1, 28, 28)
    assert dataset.y.shape == (60000, )

    assert dataset.X_train.shape == (50000, 1, 28, 28)
    assert dataset.y_train.shape == (50000, )

    assert dataset.X_valid.shape == (0, 1, 28, 28)
    assert dataset.y_valid.shape == (0, )

    assert dataset.X_test.shape == (10000, 1, 28, 28)
    assert dataset.y_test.shape == (10000, )


def test_cifar10():
    from nolearn_utils.datasets import CIFAR10
    dataset = CIFAR10()

    assert dataset.X.shape == (60000, 3, 32, 32)
    assert dataset.y.shape == (60000, )

    assert dataset.X_train.shape == (50000, 3, 32, 32)
    assert dataset.y_train.shape == (50000, )

    assert dataset.X_valid.shape == (0, 3, 32, 32)
    assert dataset.y_valid.shape == (0, )

    assert dataset.X_test.shape == (10000, 3, 32, 32)
    assert dataset.y_test.shape == (10000, )
