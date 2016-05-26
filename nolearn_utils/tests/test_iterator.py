import numpy as np
import pytest


@pytest.fixture
def rng(seed=42):
    return np.random.RandomState(seed)


@pytest.fixture
def X(rng, b=32, c=3, h=32, w=32):
    return rng.random_sample((b, c, h, w)).astype(np.float32)


@pytest.fixture
def y(rng, b=32):
    y = rng.randint(0, 10, b)
    return y.astype(np.int32)


def test_batch_iterator_mp(X, y):
    from nolearn_utils.iterators import BatchIterator
    iterator = BatchIterator(
        batch_size=6, shuffle=False,
        worker_batch_size=3, n_workers=2
    )

    n_actual = 0
    for Xb, yb in iterator(X, y):
        assert Xb.shape[1:] == (3, 32, 32)
        assert Xb.dtype == np.float32
        assert yb.dtype == np.int32
        assert len(Xb) == len(yb)
        n_actual += Xb.shape[0]

    assert n_actual == len(X)


def test_batch_iterator_mp_no_y(X):
    from nolearn_utils.iterators import BatchIterator
    iterator = BatchIterator(
        batch_size=6, shuffle=False,
        worker_batch_size=3, n_workers=2
    )

    n_actual = 0
    for Xb, yb in iterator(X):
        assert yb is None
        assert Xb.shape[1:] == (3, 32, 32)
        assert Xb.dtype == np.float32
        n_actual += Xb.shape[0]

    assert n_actual == len(X)


def test_batch_iterator_sp(X, y):
    from nolearn_utils.iterators import BatchIterator
    iterator = BatchIterator(
        batch_size=4, shuffle=False, n_workers=1
    )

    n_actual = 0
    for i, (Xb, yb) in enumerate(iterator(X, y)):
        assert Xb.shape[1:] == (3, 32, 32)
        assert Xb.dtype == np.float32
        assert yb.dtype == np.int32
        assert len(Xb) == len(yb)
        n_actual += Xb.shape[0]

    assert n_actual == len(X)


def test_add_transformer():
    # TODO
    pass


def test_transformer():
    # TODO
    pass
