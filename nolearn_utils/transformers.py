import random
from abc import abstractmethod, ABCMeta

import numpy as np
from skimage.io import imread
from skimage.exposure import equalize_hist, adjust_gamma, equalize_adapthist
from skimage.color import hsv2rgb, rgb2hsv
from skimage.color import gray2rgb as grey2rgb
from skimage.transform import resize
from .image_utils import im_affine_transform


__all__ = [
    'BaseTransformer',
    'EqualizeHistTransformer',
    'EqualizeAdaptHistTransformer',
    'KVTransformer',
    'MeanSubtractionTransformer',
    'ReadImageTransformer',
    'RandomAffineTransformer',
    'RandomAdjustGammaTransformer',
    'RandomCropTransformer',
    'RandomHorizontalFlipTransformer',
    'RandomVerticalFlipTransformer',
    'RandomAdjustHSVTransformer'
]


class BaseTransformer(object):
    """
    Base class for transformer used in the BatchIterator.
    Subclass should implement the transform_one method.
    """

    __metaclass__ = ABCMeta

    required_kws = []
    optional_kws = []

    def __init__(self, p=0.5, **kwargs):
        provided_kws = set(kwargs.keys())
        required_kws = set(self.required_kws)
        optional_kws = set(self.optional_kws)

        # Ensure there is no superfluous kws
        assert provided_kws <= (required_kws | optional_kws)

        # Ensure all required kws are provided
        assert required_kws <= provided_kws

        self.kwargs = kwargs
        self.p = p

    def __call__(self, Xb, yb=None):
        # Using a list to store the perturbed result because the transformers
        # might change the shape and the dtype of the images
        Xb_perturbed = []
        yb_perturbed = [] if yb is not None else None

        for i in range(len(Xb)):
            Xbb = Xb[i]
            ybb = yb[i] if yb is not None else None
            should_perturbed = np.random.random() < self.p

            if should_perturbed:
                kwargs = self.choose_random_kwargs()
                image_perturbed, label_perturbed = self.transform_one(Xbb, ybb, **kwargs)
                Xb_perturbed.append(image_perturbed)

                if yb is not None:
                    yb_perturbed.append(label_perturbed)
            else:
                Xb_perturbed.append(Xb[i])
                if yb is not None:
                    yb_perturbed.append(yb[i])

        Xb_perturbed = np.asarray(Xb_perturbed)
        if yb is not None:
            yb_perturbed = np.asarray(yb_perturbed)

        return Xb_perturbed, yb_perturbed

    def choose_random_kwargs(self):
        kwargs = {}
        for k, v in self.kwargs.iteritems():
            # If the method has 'rvs' property, assume this object is a
            # scipy random variable
            is_rv = hasattr(v, 'rvs')

            if is_rv:
                kwargs[k] = v.rvs(size=1)[0]
            else:
                kwargs[k] = random.choice(v)
        return kwargs

    @abstractmethod
    def transform_one(self):
        raise NotImplementedError


class EqualizeHistTransformer(BaseTransformer):
    """
    Apply histogram equalization to each image
    http://scikit-image.org/docs/dev/api/skimage.exposure.html#equalize-hist
    """

    def __init__(self, nbins=256):
        super(EqualizeHistTransformer, self).__init__(p=1.0)
        self.nbins = nbins

    def transform_one(self, img, label):
        img = equalize_hist(img, nbins=self.nbins)
        return img, label


class EqualizeAdaptHistTransformer(BaseTransformer):
    """
    Apply adaptive histogram equalization to each image
    http://scikit-image.org/docs/dev/api/skimage.exposure.html#equalize-adapthist
    """

    def __init__(self, kernel_size, clip_limit=0.01, nbins=256):
        super(EqualizeAdaptHistTransformer, self).__init__(p=1.0)
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.nbins = nbins

    def transform_one(self, img, label):
        h, w = img.shape[1:]

        img = np.asarray([
            equalize_adapthist(
                img_ch, kernel_size=self.kernel_size, clip_limit=self.clip_limit,
                nbins=self.nbins
            )
            for img_ch in img
        ])

        return img, label


class ReadImageTransformer(BaseTransformer):
    """
    Transform file name into images
    http://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imread
    """

    def __init__(self, image_size, as_grey=False):
        super(ReadImageTransformer, self).__init__(p=1.0)
        self.image_size = image_size
        self.as_grey = as_grey

    def transform_one(self, fname, label):
        img = imread(fname, as_grey=self.as_grey)
        img = resize(img, self.image_size)

        # Convert a greyscale image to rgb
        # (as_grey=False is ignored if the image is greyscale)
        if not self.as_grey and img.ndim == 2:
            img = grey2rgb(img)

        if img.ndim == 2:
            # hxw -> 1xhxw
            img = img[np.newaxis, :, :]
        else:
            # hxwx3 -> 3xhx2
            img = img.transpose(2, 0, 1)

        return img, label


class RandomAffineTransformer(BaseTransformer):
    """
    Apply random affine transformation
    """

    required_kws = []
    optional_kws = [
        'scale', 'rotation', 'shear',
        'translation_x', 'translation_y'
    ]

    def transform_one(self, img, label, **kwargs):
        img = im_affine_transform(img, **kwargs)
        return img, label


class RandomAdjustGammaTransformer(BaseTransformer):
    """
    Apply gamma adjustment at random
    http://scikit-image.org/docs/dev/api/skimage.exposure.html?highlight=local%20contrast#adjust-gamma
    """

    required_kws = ['gamma']
    optional_kws = ['gain']

    def transform_one(self, img, label, **kwargs):
        is_color = img.shape[0] == 3
        dtype = img.dtype

        if is_color:
            img = img.transpose(1, 2, 0)

        img = adjust_gamma(img, **kwargs)

        if is_color:
            img = img.transpose(2, 0, 1)

        img = img.astype(dtype)
        return img, label


class RandomCropTransformer(BaseTransformer):
    """
    Randomly crop the image to the specified size
    """

    def __init__(self, crop_size):
        super(RandomCropTransformer, self).__init__(p=1.0)
        self.crop_size = crop_size

    def transform_one(self, img, label):
        img_h = img.shape[1]
        img_w = img.shape[2]
        start_0 = np.random.choice(img_h - self.crop_size[0])
        end_0 = start_0 + self.crop_size[0]
        start_1 = np.random.choice(img_w - self.crop_size[1])
        end_1 = start_1 + self.crop_size[1]
        img = img[:, start_0:end_0, start_1:end_1].copy()
        return img, label


class RandomHorizontalFlipTransformer(BaseTransformer):
    """
    Apply random horizontal mirroring
    """

    def transform_one(self, img, label):
        img = img[:, ::-1, :]
        return img, label


class RandomVerticalFlipTransformer(BaseTransformer):
    """
    Apply random vertical mirroring
    """

    def transform_one(self, img, label):
        img = img[::-1, :, :]
        return img, label


class RandomAdjustHSVTransformer(BaseTransformer):
    """
    Apply random adjustment in HSV space
    """

    required_kws = ['h', 's', 'v']
    optional_kws = []

    def transform_one(self, img, label, h, s, v):
        assert img.shape[0] == 3
        dtype = img.dtype

        img = img.transpose(1, 2, 0)
        img_hsv = rgb2hsv(img)
        img_hsv[..., 0] = img_hsv[..., 0] + h
        img_hsv[..., 1] = img_hsv[..., 1] + s
        img_hsv[..., 2] = img_hsv[..., 2] + v
        img_hsv = img_hsv.clip(0, 1)
        img = hsv2rgb(img_hsv)
        img = img.transpose(2, 0, 1)
        img = img.astype(dtype)
        return img, label


class KVTransformer(BaseTransformer):
    """
    Look up an image using a key-value store that has a dictionary interface

    Parameters
    ----------
    mmap_image : any object with __get_item__
    mmap_label : Optional. any object with __get_item__
    """

    required_kws = []
    optional_kws = []

    def __init__(self, mmap_image, mmap_label=None):
        super(KVTransformer, self).__init__(p=1)
        self.mmap_image = mmap_image
        self.mmap_label = mmap_label

    def transform_one(self, img, label):
        img = self.mmap_image[img.tolist()]
        if self.mmap_label is not None:
            label = self.mmap_label[label.tolist()]
        return img, label


class MeanSubtractionTransformer(BaseTransformer):
    """
    Subtract predetermined mean from 
    Parameters
    ----------
    mean : numpy array. shape=(c, h, w)
    """

    required_kws = []
    optional_kws = []

    def __init__(self, mean):
        self.mean = mean
        super(MeanSubtractionTransformer, self).__init__(p=1)

    def transform_one(self, img, label):
        img = img - self.mean
        return img, label
