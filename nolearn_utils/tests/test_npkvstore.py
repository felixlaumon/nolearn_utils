import pytest
import numpy as np
from nolearn_utils.npkvstore import NPKVStore
from nolearn_utils.utils import mkdirp


@pytest.yield_fixture
def kv_store():
    mkdirp('/tmp/nolearn_utils/')
    kv_store = NPKVStore(
        '/tmp/nolearn_utils/npkvstore.dat',
        dtype=np.float32, mode='w+', shape=(100)
    )
    yield kv_store
    kv_store.destroy()


@pytest.yield_fixture
def kv_store_rplus():
    mkdirp('/tmp/nolearn_utils/')
    kv_store = NPKVStore(
        '/tmp/nolearn_utils/npkvstore.dat',
        dtype=np.float32, mode='r+', shape=(100)
    )
    yield kv_store


def test_npkvstore_get_set_single_item(kv_store):
    kv_store['a', 1] = 42

    assert kv_store['a', 1] == 42


def test_npkvstore_reuse_index(kv_store):
    kv_store['a', 1] = 1
    kv_store['a', 1] = 2

    assert kv_store['a', 1] == 2


def test_npkvstore_flush(kv_store, kv_store_rplus):
    kv_store['a', 1] = 1
    kv_store['a', 2] = 2

    assert ('a', 1) not in kv_store_rplus
    assert ('a', 2) not in kv_store_rplus

    kv_store.flush()
    kv_store_rplus.reload()

    assert ('a', 1) in kv_store_rplus
    assert ('a', 2) in kv_store_rplus


def test_npkvsotere_write_more(kv_store, kv_store_rplus):
    kv_store['a', 1] = 1
    kv_store.flush()

    kv_store_rplus.reload()
    kv_store_rplus['a', 2] = 2

    assert kv_store_rplus['a', 1] == 1
    assert kv_store_rplus['a', 2] == 2
