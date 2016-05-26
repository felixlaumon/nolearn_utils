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


@pytest.fixture(scope='session')
def coffee_image_fname(tmpdir_factory):
    from skimage.data import coffee
    from skimage.io import imsave
    img = coffee()
    fn = tmpdir_factory.mktemp('data').join('img.png')
    imsave(str(fn), img)
    return fn


def test_equalize_hist(X):
    from nolearn_utils.transformers import EqualizeHistTransformer
    tformer = EqualizeHistTransformer()
    tformer(X)


def test_equalize_adapthist(X):
    from nolearn_utils.transformers import EqualizeAdaptHistTransformer
    tformer = EqualizeAdaptHistTransformer(
        kernel_size=(3, 3)
    )
    tformer(X)


def test_read_image_transformer(coffee_image_fname):
    from nolearn_utils.transformers import ReadImageTransformer

    tformer = ReadImageTransformer(image_size=(32, 32))
    Xt, _ = tformer([coffee_image_fname, coffee_image_fname])
    assert Xt.shape == (2, 3, 32, 32)


def test_random_affine_transformer(X):
    from scipy.stats import uniform
    from nolearn_utils.transformers import RandomAffineTransformer

    tformer = RandomAffineTransformer(
        scale=uniform(0.75, 0.5),
        rotation=uniform(0, 360),
        shear=uniform(-0.1, 0.2),
        translation_x=[-2, 0, 2],
        translation_y=[-2, 0, 2]
    )
    tformer(X)


def test_random_adjust_gamma_transformer(X):
    from nolearn_utils.transformers import RandomAdjustGammaTransformer

    tformer = RandomAdjustGammaTransformer(
        gamma=[0.75, 1, 1.25]
    )
    tformer(X)


def test_random_crop_transformer(X):
    from nolearn_utils.transformers import RandomCropTransformer

    tformer = RandomCropTransformer(crop_size=(30, 30))
    tformer(X)


def test_random_horizontal_flip_transformer(X):
    from nolearn_utils.transformers import RandomHorizontalFlipTransformer

    tformer = RandomHorizontalFlipTransformer()
    tformer(X)


def test_random_vertical_flip_transformer(X):
    from nolearn_utils.transformers import RandomVerticalFlipTransformer

    tformer = RandomVerticalFlipTransformer()
    tformer(X)


def test_random_adjust_hsv_transformer(X):
    from scipy.stats import norm
    from nolearn_utils.transformers import RandomAdjustHSVTransformer

    tformer = RandomAdjustHSVTransformer(
        h=norm(loc=1, scale=0.1),
        s=norm(loc=1, scale=0.1),
        v=norm(loc=1, scale=0.1)
    )
    tformer(X)


def test_mean_subtraction_transformer(X):
    from nolearn_utils.transformers import MeanSubtractionTransformer

    mean = np.random.random((3, 32, 32))
    tformer = MeanSubtractionTransformer(mean=mean)
    tformer(X)
