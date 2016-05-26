"""
Useful group of layers
"""
from lasagne.layers import DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPoolLayer
from lasagne.layers.normalization import batch_norm


def vgg(l, name, num_layers, downsample='maxpool', drop_p=0, bn=True,
        **kwargs):
    """Returns a VGG layer group

    Parameters
    ----------
    name : str
        Prefix for layer name

    num_layers : int
        Number of convolutional layers

    downsample : str, default='maxpool'
        'maxpool', or 'conv_stride'

    drop_p : float
        0 to 1

    bn : bool, default=True
        If True, apply batch normalization
    """
    assert downsample in ['maxpool', 'stride']

    for i in range(num_layers):
        conv_kwargs = {}
        conv_kwargs.update(kwargs)

        if (downsample == 'stride') and (i == (num_layers - 1)):
            conv_kwargs['stride'] = 2

        l = ConvLayer(
            l, name='%sc%s' % (name, i + 1),
            **conv_kwargs
        )
        if bn:
            l = batch_norm(l, name='%sc%sbn' % (name, i + 1))

    if drop_p > 0:
        l = DropoutLayer(l, name='%sdrop' % name, p=drop_p)

    if downsample == 'maxpool':
        l = MaxPoolLayer(l, name='%spool' % name, pool_size=2)

    return l


def residual_block():
    pass
