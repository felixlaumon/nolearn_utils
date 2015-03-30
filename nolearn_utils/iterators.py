from __future__ import division
from __future__ import print_function

import sys
from time import time

import numpy as np
from numpy.random import choice
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.transform import resize
from skimage.transform import SimilarityTransform
from skimage.transform import AffineTransform
from skimage.transform import warp


class BaseBatchIterator(object):
    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size
        self.verbose = kwargs.get('verbose', False)

    def __call__(self, X, y=None, verbose=False):
        # TODO sanity check if X and y has same length
        self.X = X
        self.y = y
        return self

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        n_iterations = (n_samples + bs - 1) // bs
        for i in range(n_iterations):
            t0 = time()
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)
            write_temp_log('Minibatch %i / %i (%.2fs)'
                           % (i + 1, n_iterations, time() - t0))
        write_temp_log('')  # Remove last log

    def transform(self, Xb, yb):
        return Xb, yb


class ShuffleBatchIteratorMixin(object):
    """
    From https://github.com/dnouri/nolearn/issues/27#issuecomment-71175381
    """
    def __iter__(self):
        self.X, self.y = shuffle(self.X, self.y)
        for res in super(ShuffleBatchIteratorMixin, self).__iter__():
            yield res


class AffineTransformBatchIteratorMixin(object):
    def __init__(self, affine_p,
                 affine_scale_choices, affine_translation_choices,
                 affine_rotation_choices,
                 *args, **kwargs):
        super(AffineTransformBatchIteratorMixin,
              self).__init__(*args, **kwargs)
        self.affine_p = affine_p
        self.affine_scale_choices = affine_scale_choices
        self.affine_translation_choices = affine_translation_choices
        self.affine_rotation_choices = affine_rotation_choices

        if self.verbose:
            print('Random transform probability: %.2f' % self.affine_p)
            print('Rotation choices', self.affine_rotation_choices)
            print('Scale choices', self.affine_scale_choices)
            print('Translation choices', self.affine_translation_choices)

    def transform(self, Xb, yb):
        Xb, yb = super(AffineTransformBatchIteratorMixin,
                       self).transform(Xb, yb)
        # Skip if affine_p is 0. Setting affine_p may be useful for quikcly
        # disabling affine transformation
        if self.affine_p == 0:
            return Xb, yb
        idx = get_random_idx(Xb, self.affine_p)
        Xb_transformed = np.empty_like(Xb)
        for i, img in enumerate(Xb):
            scale = choice(self.affine_scale_choices)
            rotation = choice(self.affine_rotation_choices)
            translation_y = choice(self.affine_translation_choices)
            translation_x = choice(self.affine_translation_choices)
            img_transformed = im_affine_transform(img, scale=scale,
                                                  rotation=rotation,
                                                  translation_y=translation_y,
                                                  translation_x=translation_x)
            Xb_transformed[i] = img_transformed
        return Xb, yb


class RandomCropBatchIteratorMixin(object):
    def __init__(self, crop_size, *args, **kwargs):
        super(RandomCropBatchIteratorMixin, self).__init__(*args, **kwargs)
        self.crop_size = crop_size

    def transform(self, Xb, yb):
        Xb, yb = super(RandomCropBatchIteratorMixin, self).transform(Xb, yb)
        batch_size = min(self.batch_size, Xb.shape[0])
        img_h = Xb.shape[2]
        img_w = Xb.shape[3]
        Xb_transformed = np.empty((batch_size, Xb.shape[1],
                                   self.crop_size[0], self.crop_size[1]))
        # TODO vectorize implementation if possible
        for i in range(batch_size):
            start_0 = np.random.choice(img_h - self.crop_size[0])
            end_0 = start_0 + self.crop_size[0]
            start_1 = np.random.choice(img_w - self.crop_size[1])
            end_1 = start_1 + self.crop_size[1]
            Xb_transformed[i] = Xb[i][:, start_0:end_0, start_1:end_1]
        return Xb_transformed, yb


class ReadImageBatchIteratorMixin(object):
    def __init__(self, read_image_size, read_image_prefix_path='',
                 read_image_as_gray=False, read_image_as_bc01=True,
                 read_image_as_float32=True,
                 *args, **kwargs):
        super(ReadImageBatchIteratorMixin, self).__init__(*args, **kwargs)
        self.read_image_size = read_image_size
        self.read_image_prefix_path = read_image_prefix_path
        self.read_image_as_gray = read_image_as_gray
        self.read_image_as_bc01 = read_image_as_bc01
        self.read_image_as_float32 = read_image_as_float32

    def transform(self, Xb, yb):
        Xb, yb = super(ReadImageBatchIteratorMixin, self).transform(Xb, yb)

        batch_size = min(Xb.shape[0], self.batch_size)
        num_channels = 1 if self.read_image_as_gray is True else 3
        h = self.read_image_size[0]
        w = self.read_image_size[1]

        imgs = np.empty((batch_size, num_channels, h, w), dtype=np.float32)
        for i, path in enumerate(Xb):
            img = imread(self.read_image_prefix_path + path,
                         as_gray=self.read_image_as_gray)
            img = resize(img, (h, w))

            # Convert greyscale image to RGB for consistency
            if len(img.shape) == 2 and self.read_image_as_gray is False:
                img = gray2rgb(img)

            # Transpose to bc01
            if self.read_image_as_bc01 and self.read_image_as_gray is False:
                img = img.transpose(2, 0, 1)
            elif self.read_image_as_bc01 and self.read_image_as_gray is True:
                img = np.expand_dims(img, axis=0)

            imgs[i] = img
        return imgs, yb


class MeanSubtractBatchiteratorMixin(object):
    def __init__(self, mean):
        pass

    def transform(self, Xb, yb):
        pass


def shuffle(*arrays):
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]


def im_affine_transform(img, scale, rotation, translation_y, translation_x):
    img = img.transpose(1, 2, 0)
    # Normalize so that the param acts more like im_rotate, im_translate etc
    scale = 1 / scale
    translation_x = - translation_x

    # shift to center first so that image is rotated around center
    center_shift = np.array((img.shape[0], img.shape[1])) / 2. - 0.5
    tform_center = SimilarityTransform(translation=-center_shift)
    tform_uncenter = SimilarityTransform(translation=center_shift)

    rotation = np.deg2rad(rotation)
    tform = AffineTransform(scale=(scale, scale), rotation=rotation,
                            translation=(translation_y, translation_x))
    tform = tform_center + tform + tform_uncenter

    warped_img = warp(img, tform)
    warped_img = warped_img.transpose(2, 0, 1)
    return warped_img


def get_random_idx(arr, p):
    n = arr.shape[0]
    idx = choice(n, int(n * p), replace=False)
    return idx


def write_temp_log(str):
    sys.stdout.write('\r%s' % str)
    sys.stdout.flush()
