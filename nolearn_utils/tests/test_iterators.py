import numpy as np
import pytest
from nolearn_utils.iterators import BaseBatchIterator


@pytest.fixture
def X():
    return np.random.rand(1000, 3, 48, 48)


@pytest.fixture
def y():
    return np.random.rand(1000)


def test_base_batch_iterator(X, y):
    iterator = BaseBatchIterator(batch_size=128)
    batches = list(iterator(X, y))
    assert len(batches) == 8
    for Xb, yb in batches[:-1]:
        assert Xb.shape[0] == 128
        assert yb.shape[0] == 128
    assert batches[-1][0].shape[0] == 104
    assert batches[-1][1].shape[0] == 104


def test_shuffle_batch_iterator(X, y):
    from nolearn_utils.iterators import ShuffleBatchIteratorMixin

    class Iterator(ShuffleBatchIteratorMixin, BaseBatchIterator):
        pass
    iterator = Iterator(batch_size=128)
    np.random.seed(42)  # Deterministic tests
    Xb, yb = iter(iterator(X, y)).next()
    assert np.all(Xb != X[:128]) == True
    assert np.all(yb != y[:128]) == True
