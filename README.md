# nolearn-utils

Iterators and hooks for nolearn.lasagne

## Iterators

````py
class TrainIterator(RandomCropBatchIteratorMixin,
                   AffineTransformBatchIteratorMixin,
                   ShuffleBatchIteratorMixin,
                   BaseBatchIterator):
    pass

iterator = TrainIterator(batch_size=128, verbose=True,
                        affine_p=0.25,
                        affine_translation_choices=np.arange(0, 1),
                        affine_scale_choices=np.linspace(0.9, 1.1, 5),
                        affine_rotation_choices=np.arange(0, 360, 15),
                        crop_size=(300, 300))
```

## Hooks

...

## License

MIT & BSD
