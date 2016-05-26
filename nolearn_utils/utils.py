"""
Other useful utilities for nolearn_utils
"""
import os
from importlib import import_module


class PredeterminedTrainSplit(object):
    """
    Force the training and validation set

    The training set passed to fit(...) will be ignored
    """
    def __init__(self, X_train, X_valid, y_train, y_valid):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def __call__(self, X, y, net):
        return self.X_train, self.X_valid, self.y_train, self.y_valid


def get_model(name, package=None):
    """Load a model definition"""
    model = import_module(name, package)

    assert hasattr(model, 'get_net')
    assert hasattr(model, 'get_layers')
    assert hasattr(model, 'get_iterators')

    return model


def mkdirp(dirname):
    """Make directory if not available"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
