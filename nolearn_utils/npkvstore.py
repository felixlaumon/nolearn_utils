import os
import cPickle as pickle
import numpy as np


def add_suffix(path, suffix):
    """Add suffix to the file name of the path"""
    has_leading_slash = '/' in path
    parts = path.split(os.path.sep)
    dirname = os.path.join(*parts[:-1])
    fname_ext = parts[-1]
    fname, ext = os.path.splitext(fname_ext)

    fname = fname + suffix

    fname_ext = fname + ext
    path = os.path.join(dirname, fname_ext)

    if has_leading_slash:
        path = '/' + path

    return path


class NPKVStore(object):
    """Key value store backed by numpy memmap"""

    def __init__(self, mmap_fname, dtype, mode, shape):
        self.mmap_fname = mmap_fname
        self.key_fname = add_suffix(mmap_fname, '_idx')
        self.dtype = dtype
        self.mode = mode
        self.shape = shape

        self.reload()

    def __repr__(self):
        return self.mmap.__repr__()

    def _key_to_index(self, key):
        return self.keys[key]

    def _keys_to_indices(self, keys):
        return [self._key_to_index(key) for key in keys]

    def _add_key(self, key):
        assert key not in self
        self.keys[key] = self._index
        self._index += 1

    def _remove_key(self, key):
        del self.keys[key]

    def __getitem__(self, keys):
        if type(keys) == list:
            return self.mmap[self._keys_to_indices(keys)]
        else:
            key = keys
            # http://stackoverflow.com/questions/18614927/how-to-slice-memmap-efficiently
            # TODO slicing with list copy the whole thing to memory
            return self.mmap[self._key_to_index(key)]

    def __setitem__(self, keys, values):
        if type(keys) == list:
            for key in keys:
                if key not in self:
                    self._add_key(key)

            self.mmap[self._keys_to_indices(keys)] = values
        else:
            key = keys
            if key not in self:
                self._add_key(key)
            self.mmap[self._key_to_index(key)] = values

    def __contains__(self, key):
        return key in self.keys

    def remove(self, key):
        # TODO can only remove single key at a time
        self.mmap[self._key_to_index(key)] = 0
        self._remove_key(key)

    def flush(self):
        self.mmap.flush()
        pickle.dump(
            self.keys, open(self.key_fname, self.mode),
            pickle.HIGHEST_PROTOCOL
        )

    def destroy(self):
        if os.path.exists(self.mmap_fname):
            os.remove(self.mmap_fname)

        if os.path.exists(self.key_fname):
            os.remove(self.key_fname)

    def reload(self):
        self.mmap = np.memmap(
            self.mmap_fname,
            dtype=self.dtype, mode=self.mode, shape=self.shape
        )

        # TODO not shareable among processes
        if os.path.exists(self.key_fname) and self.mode != 'w+':
            self.keys = pickle.load(open(self.key_fname, self.mode))
        else:
            self.keys = {}

        if len(self.keys) > 0:
            self._index = max([v for v in self.keys.values()]) + 1
        else:
            self._index = 0
