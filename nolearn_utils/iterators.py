from __future__ import division
from __future__ import print_function

import sys
import os
from time import time

import numpy as np
from numpy.random import choice
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.transform import resize
from skimage.transform import SimilarityTransform
from skimage.transform import AffineTransform
from skimage.transform import warp
from skimage.filters import rank
from skimage.morphology import disk


class BaseBatchIterator(object):
    def __init__(self, batch_size, verbose=False):
        self.batch_size = batch_size
        self.verbose = False

    def __call__(self, X, y=None):
        self.X, self.y = X, y
        return self

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('X', 'y',):
            if attr in state:
                del state[attr]
        return state


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
        # TODO raise exception if Xb size is smaller than crop size
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


class RandomFlipBatchIteratorMixin(object):
    def __init__(self, flip_horizontal_p=0.5, flip_vertical_p=0.5, *args, **kwargs):
        super(RandomFlipBatchIteratorMixin, self).__init__(*args, **kwargs)
        self.flip_horizontal_p = flip_horizontal_p
        self.flip_vertical_p = flip_vertical_p

    def transform(self, Xb, yb):
        Xb, yb = super(RandomFlipBatchIteratorMixin, self).transform(Xb, yb)
        Xb_flipped = Xb.copy()

        if self.flip_horizontal_p > 0:
            horizontal_flip_idx = get_random_idx(Xb, self.flip_horizontal_p)
            Xb_flipped[horizontal_flip_idx] = Xb_flipped[horizontal_flip_idx, :, :, ::-1]

        if self.flip_vertical_p > 0:
            vertical_flip_idx = get_random_idx(Xb, self.flip_vertical_p)
            Xb_flipped[vertical_flip_idx] = Xb_flipped[vertical_flip_idx, :, ::-1, :]

        return Xb_flipped, yb


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
            img_fname = os.path.join(self.read_image_prefix_path, path)
            img = imread(img_fname,
                         as_grey=self.read_image_as_gray)
            img = resize(img, (h, w))

            # When reading image as color image, convert grayscale image to RGB for consistency
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
    """
    TODO should calculate the mean
    """
    def __init__(self, mean, *args, **kwargs):
        super(MeanSubtractBatchiteratorMixin, self).__init__(*args, **kwargs)
        self.mean = mean

    def transform(self, Xb, yb):
        Xb, yb = super(MeanSubtractBatchiteratorMixin, self).transform(Xb, yb)
        Xb = Xb - self.mean
        return Xb, yb


class LCNBatchIteratorMixin(object):
    """
    """
    def __init__(self, lcn_selem=disk(5), *args, **kwargs):
        super(LCNBatchIteratorMixin, self).__init__(*args, **kwargs)
        self.lcn_selem = lcn_selem

    def transform(self, Xb, yb):
        Xb, yb = super(LCNBatchIteratorMixin, self).transform(Xb, yb)
        if len(Xb.shape) != 4 or Xb.shape[1] != 1:
            raise ValueError('X must be in shape of (batch_size, 1, height, width) but is %s', Xb.shape)
        Xb_transformed = np.asarray([local_contrast_normalization(img[0], self.lcn_selem) for img in Xb])
        Xb_transformed = Xb_transformed[:, np.newaxis, :, :]
        return Xb_transformed, yb


def make_iterator(name, mixin):
    """
    Return an Iterator class added with the provided mixin
    """
    mixin = [BaseBatchIterator] + mixin
    # Reverse the order for type()
    mixin.reverse()
    return type(name, tuple(mixin), {})


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


def local_contrast_normalization(img, selem=disk(5)):
    return rank.equalize(img, selem)


def get_random_idx(arr, p):
    n = arr.shape[0]
    idx = choice(n, int(n * p), replace=False)
    return idx


def write_temp_log(str):
    sys.stdout.write('\r%s' % str)
    sys.stdout.flush()
