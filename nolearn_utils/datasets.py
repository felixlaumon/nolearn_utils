from abc import ABCMeta
import os
import requests
import urlparse
from itertools import izip

import numpy as np
from tqdm import tqdm

from nolearn_utils.utils import mkdirp
from nolearn_utils.npkvstore import NPKVStore

DIRNAME = '~/.nolearn_utils/datasets'
DIRNAME = os.path.expanduser(DIRNAME)


def download(url, fname):
    """Download a URL to a directory"""
    file_size = int(requests.head(url).headers['content-length'])
    res = requests.get(url, stream=True)
    gen = tqdm(
        res.iter_content(), desc=url,
        unit="B", unit_scale=True, total=file_size
    )
    # TODO ensure fname does not exist
    with open(fname, 'wb') as f:
        for data in gen:
            f.write(data)


class BaseDataset(object):
    __metaclass__ = ABCMeta

    urls = {}

    X_dtype = np.float32
    y_dtype = np.int32

    X_shape = None
    y_shape = None

    def __init__(self, dirname=DIRNAME):
        assert isinstance(self.X_shape, tuple)
        assert isinstance(self.y_shape, tuple)

        self.dirname = os.path.join(dirname, self.name)
        mkdirp(self.dirname)

        self.X_mmap_fname = os.path.join(self.dirname, 'X.dat')
        self.y_mmap_fname = os.path.join(self.dirname, 'y.dat')

        self.train_idx_fname = os.path.join(self.dirname, 'train_idx.npy')
        self.valid_idx_fname = os.path.join(self.dirname, 'valid_idx.npy')
        self.test_idx_fname = os.path.join(self.dirname, 'test_idx.npy')

        if not self.downloaded:
            self.download()

        if not self.cached:
            self.cache()

        self.open_cache(mode='r')
        self.open_idx()

    def download(self):
        for file_tag, url in self.urls.iteritems():
            download(url, self.get_fname(file_tag))

    def get_fname(self, file_tag):
        url = self.urls[file_tag]
        fname = url.split('/')[-1]
        return os.path.join(self.dirname, fname)

    def open_cache(self, mode='r'):
        self.X_mmap = NPKVStore(
            self.X_mmap_fname, dtype=self.X_dtype,
            mode=mode, shape=self.X_shape
        )

        # TODO y should be optional
        self.y_mmap = NPKVStore(
            self.y_mmap_fname, dtype=self.y_dtype,
            mode=mode, shape=self.y_shape
        )

    def open_idx(self):
        self.train_idx = np.load(self.train_idx_fname).tolist()
        self.valid_idx = np.load(self.valid_idx_fname).tolist()
        self.test_idx = np.load(self.test_idx_fname).tolist()

    def cache(self):
        self.open_cache(mode='w+')

        train_idx = []
        valid_idx = []
        test_idx = []

        for key, data, target in tqdm(self.get_train_data()):
            self.X_mmap[key] = data
            self.y_mmap[key] = target
            train_idx.append(key)

        self.X_mmap.flush()
        self.y_mmap.flush()

        for key, data, target in tqdm(self.get_valid_data()):
            self.X_mmap[key] = data
            self.y_mmap[key] = target
            valid_idx.append(key)

        self.X_mmap.flush()
        self.y_mmap.flush()

        for key, data, target in tqdm(self.get_test_data()):
            self.X_mmap[key] = data
            self.y_mmap[key] = target
            test_idx.append(key)

        self.X_mmap.flush()
        self.y_mmap.flush()

        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx

        self.cache_idx()

    def cache_idx(self):
        np.save(self.train_idx_fname, self.train_idx)
        np.save(self.valid_idx_fname, self.valid_idx)
        np.save(self.test_idx_fname, self.test_idx)

    def get_train_data(self):
        return []

    def get_valid_data(self):
        return []

    def get_test_data(self):
        return []

    @property
    def cached(self):
        if not os.path.exists(self.X_mmap_fname):
            return False
        if not os.path.exists(self.y_mmap_fname):
            return False
        return True

    @property
    def downloaded(self):
        for file_tag, _ in self.urls.iteritems():
            is_exist = os.path.exists(self.get_fname(file_tag))
            if not is_exist:
                return False
        return True

    @property
    def X(self):
        return self.X_mmap

    @property
    def y(self):
        return self.y_mmap

    @property
    def X_train(self):
        return self.X[self.train_idx]

    @property
    def X_test(self):
        return self.X[self.test_idx]

    @property
    def X_valid(self):
        return self.X[self.valid_idx]

    @property
    def y_train(self):
        return self.y[self.train_idx]

    @property
    def y_valid(self):
        return self.y[self.valid_idx]

    @property
    def y_test(self):
        return self.y[self.test_idx]


class MNIST(BaseDataset):
    name = 'mnist'

    X_shape = (60000, 1, 28, 28)
    y_shape = (60000, )

    def get_train_data(self):
        from sklearn.datasets import fetch_mldata
        from sklearn.cross_validation import train_test_split

        mnist = fetch_mldata('MNIST original')
        X_train, _, y_train, _ = train_test_split(
            mnist['data'], mnist['target'],
            train_size=50000, random_state=42, stratify=mnist['target']
        )
        X_train = X_train.reshape(-1, 1, 28, 28) / 255.0
        return izip(range(50000), X_train, y_train)

    def get_test_data(self):
        from sklearn.datasets import fetch_mldata
        from sklearn.cross_validation import train_test_split

        mnist = fetch_mldata('MNIST original')
        X_test, _, y_test, _ = train_test_split(
            mnist['data'], mnist['target'],
            test_size=10000, random_state=42, stratify=mnist['target']
        )
        X_test = X_test.reshape(-1, 1, 28, 28) / 255.0
        return izip(range(10000), X_test, y_test)


class CIFAR10(BaseDataset):
    name = 'cifar10'

    urls = {
        'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    }

    X_shape = (60000, 3, 32, 32)
    y_shape = (60000, )

    def download(self):
        import tarfile

        super(self, CIFAR10).download()

        self.open_cache(mode='w+')
        fname = self.get_fname('cifar10')
        tar = tarfile.open(fname)
        tar.extractall()
        tar.close()

    def get_train_data(self):
        import cPickle as pickle

        for batch_idx in range(1, 6):
            fname = os.path.join(
                self.dirname,
                'cifar-10-batches-py',
                'data_batch_%s' % batch_idx
            )
            d = pickle.load(open(fname, 'rb'))
            filenames = d['filenames']
            data = d['data'].reshape(-1, 3, 32, 32)
            data = data / 255.0
            targets = d['labels']

            for train_data in izip(filenames, data, targets):
                yield train_data

    def get_test_data(self):
        import cPickle as pickle

        fname = os.path.join(
            self.dirname,
            'cifar-10-batches-py',
            'test_batch'
        )
        d = pickle.load(open(fname, 'rb'))
        filenames = d['filenames']
        data = d['data'].reshape(-1, 3, 32, 32)
        data = data / 255.0
        targets = d['labels']

        for test_data in izip(filenames, data, targets):
            yield test_data

if __name__ == '__main__':
    mnist = MNIST()
    # cifar10 = CIFAR10()
