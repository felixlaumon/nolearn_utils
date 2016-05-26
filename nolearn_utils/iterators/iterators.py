import uuid
import itertools
from multiprocessing import Process, Queue, cpu_count
import numpy as np
import SharedArray as sa


__all__ = [
    'BatchIterator'
]


class BatchIterator(object):
    """Real-time augmentation using multiple processes
    """

    def __init__(self, batch_size, worker_batch_size=None, n_workers=1,
                 shuffle=True):
        if worker_batch_size is None:
            worker_batch_size = batch_size

        if n_workers == 'all':
            n_workers = cpu_count()

        if batch_size % worker_batch_size != 0:
            raise NotImplementedError(
                'batch_size must be multiple of worker_batch_size'
            )

        if n_workers == 1 and worker_batch_size != batch_size:
            raise NotImplementedError(
                'batch_size must be the same as worker_batch_size when'
                'n_workers is 1'
            )

        self.batch_size = batch_size
        self.worker_batch_size = worker_batch_size

        self.n_workers = n_workers
        self.shuffle = shuffle

        self.pre_dispatch = 2 * self.n_workers
        self.transformers = []

        self.uuid = None
        self.X = None
        self.y = None
        self.idx = None

    @property
    def n_samples(self):
        """
        Returns the number of injected samples
        """
        if self.X is None:
            raise ValueError('X have not been injected')
        return self.X.shape[0]

    @property
    def n_batches(self):
        """
        Returns the number of batches
        """
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    @property
    def n_worker_batches(self):
        """
        Returns the number of batches for the worker
        """
        return (
            (self.n_samples + self.worker_batch_size - 1) //
            self.worker_batch_size
        )

    def add_transformer(self, transformer):
        """
        Add transformer

        Parameters
        ----------
        transformer : instance of subclass of
                      nolearn_utils.transformer.BaseTransformer
        """
        self.transformers.append(transformer)

    def __call__(self, X, y=None):
        """
        Inject X and y into the iterator

        Parameters
        ----------
        X : numpy array
        y : numpy array
        """
        if y is not None:
            assert len(X) == len(y)
        self.X = X
        self.y = y

        self.idx = np.arange(len(X))
        if self.shuffle:
            np.random.shuffle(self.idx)

        return self

    def __iter__(self):
        if self.n_workers == 1:
            for item in self._sp_iter():
                yield item
        else:
            for item in self._mp_iter():
                yield item

    def _sp_iter(self):
        for batch_idx in range(self.n_worker_batches):
            yield self._get_origingal_batch(batch_idx)

    def _mp_iter(self):
        self.uuid = uuid.uuid4()

        def _enqueue(in_queue, n_batches, n_workers):
            for batch_idx in range(n_batches):
                # print('enqueue', batch_idx)
                in_queue.put(batch_idx)
            in_queue.close()

        def _transform(in_queue, out_queue):
            for batch_idx in in_queue:
                # print('transform', batch_idx)
                Xb, yb = self._get_origingal_batch(batch_idx)
                Xb, yb = self._apply_transformers(Xb, yb)
                self._set_transformed_batch(batch_idx, Xb, yb)
                # print('done', batch_idx)
                out_queue.put(batch_idx)
            out_queue.close()

        self._job_queue = IterableQueue(maxsize=self.pre_dispatch)
        self._done_queue = IterableQueue(
            maxsize=self.pre_dispatch, close_count=self.n_workers
        )

        enqueue_process = Process(
            target=_enqueue, args=(
                self._job_queue, self.n_worker_batches, self.n_workers
            )
        )
        transform_processes = [
            Process(target=_transform, args=(
                self._job_queue, self._done_queue
            ))
            for _ in range(self.n_workers)
        ]

        for p in transform_processes:
            p.start()
        enqueue_process.start()

        k = self.batch_size // self.worker_batch_size
        for batch_idxes in combine_chunks(self._done_queue, k=k):
            Xbs = []
            ybs = []
            for batch_idx in batch_idxes:
                Xb, yb = self._get_transformed_batch(batch_idx)
                Xbs.append(Xb)
                ybs.append(yb)

            Xbs = np.concatenate(Xbs)
            if self.y is not None:
                ybs = np.concatenate(ybs)
                yield Xbs, ybs
            else:
                yield Xbs, None

        for p in transform_processes:
            p.join()
        enqueue_process.join()
        self.uuid = None

    def _get_origingal_batch(self, worker_batch_idx):
        start = worker_batch_idx * self.worker_batch_size
        end = (worker_batch_idx + 1) * self.worker_batch_size

        idx = self.idx[start:end]
        Xb = self.X[idx]

        if self.y is not None:
            yb = self.y[idx]
        else:
            yb = None

        return Xb, yb

    def _set_transformed_batch(self, batch_id, Xb, yb=None):
        Xb_name = '%s_%s_Xb' % (self.uuid, batch_id)
        Xb_sa = sa.create(Xb_name, shape=Xb.shape, dtype=Xb.dtype)
        Xb_sa[:] = Xb[:]

        if yb is not None:
            yb_name = '%s_%s_yb' % (self.uuid, batch_id)
            yb_sa = sa.create(yb_name, shape=yb.shape, dtype=yb.dtype)
            yb_sa[:] = yb[:]

    def _get_transformed_batch(self, batch_id):
        Xb_name = '%s_%s_Xb' % (self.uuid, batch_id)
        Xb = sa.attach(Xb_name)
        sa.delete(Xb_name)

        if self.y is not None:
            yb_name = '%s_%s_yb' % (self.uuid, batch_id)
            yb = sa.attach(yb_name)
            sa.delete(yb_name)
        else:
            yb = None

        return Xb, yb

    def _apply_transformers(self, Xb, yb):
        for transformer in self.transformers:
            Xb, yb = transformer(Xb, yb)
        return Xb, yb

    def __del__(self):
        names = [
            name for name in sa.list() if self.uuid in name
        ]

        for name in names:
            sa.delete(name)


class ResampleBatchIterator(BatchIterator):
    """
    Oversample or undersample the training samples using the provided
    per-label weights
    """
    def __init__(self, rebalance_weights, *args, **kwargs):
        self.rebalance_weights = rebalance_weights
        super(ResampleBatchIterator, self).__init__(*args, **kwargs)

    def __call__(self, X, y=None):
        if y is None:
            raise ValueError('ResampleBatchIterator requires y')

        super(ResampleBatchIterator, self).__call__(X, y)

        assert y.ndim == 1
        n = len(X)
        ydist = np.bincount(y).astype(float) / len(y)
        idx = np.arange(n)

        # Create sampling probablity list based on the target per-label weights
        p = np.zeros_like(idx, dtype=float)
        for dist, (label, target_dist) in zip(ydist, enumerate(self.rebalance_weights)):
            p[y == label] = target_dist / dist
        p /= p.sum()

        idx = np.random.choice(idx, size=n, p=p)
        self.idx = idx
        return self


class RandomSliceBatchIterator(BatchIterator):
    """
    """

    def __init__(self, n, interval, batch_size, random_shift=True,
                 worker_batch_size=None, n_workers=1, shuffle=False, seed=42):
        self.n = n
        self.interval = interval
        self.random_shift = random_shift
        super(RandomSliceBatchIterator, self).__init__(
            batch_size, worker_batch_size=worker_batch_size,
            n_workers=n_workers, shuffle=shuffle, seed=seed
        )

    def __call__(self, X, y=None):
        X = np.array([
            self.random_slice(
                x, n=self.n, interval=self.interval,
                random_shift=self.random_shift
            )
            for x in X
        ])
        return super(RandomSliceBatchIterator, self).__call__(X, y=y)

    def random_slice(self, el, n, interval=1, random_shift=True):
        """
        """
        actual_n = n * interval
        start = np.random.choice(len(el) - actual_n) if random_shift else 0
        end = start + actual_n
        assert end <= len(el) - 1
        el = list(itertools.islice(el, start, end, interval))
        return el


class IterableQueue(object):
    """
    """

    def __init__(self, maxsize, close_count=1):
        self.queue = Queue(maxsize=maxsize)
        self.close_count = close_count

    def __iter__(self):
        n_sentinel = 0
        while True:
            item = self.queue.get()
            if item is None:
                n_sentinel += 1
                if n_sentinel >= self.close_count:
                    self.queue.put(None)
                    return
                continue
            else:
                yield item

    def close(self):
        self.put(None)

    def put(self, item):
        self.queue.put(item)


def combine_chunks(it, k):
    items = []

    for item in it:
        items.append(item)

        if len(items) == k:
            yield items
            items = []

    if len(items) > 0:
        yield items
