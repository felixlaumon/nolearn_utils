import numpy as np
import pandas as pd

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates

from nolearn.lasagne import NeuralNet
from nolearn.lasagne.handlers import SaveWeights, PrintLayerInfo
from nolearn.lasagne.util import get_conv_infos

from nolearn_utils.iterators import (
    BufferedBatchIteratorMixin,
    ReadImageBatchIteratorMixin,
    ShuffleBatchIteratorMixin,
    LCNBatchIteratorMixin,
    RandomFlipBatchIteratorMixin,
    EqualizeAdaptHistBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    make_iterator
)
from nolearn_utils.hooks import SaveTrainingHistory, PlotTrainingHistory


image_size = 28
mnist = fetch_mldata('MNIST original')
X = mnist.data.reshape(-1, 1, image_size, image_size).astype(np.float32) / 255
y = mnist.target.astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

batch_size = 256
n_classes = 10

train_iterator_mixins = [
    ShuffleBatchIteratorMixin,
    RandomFlipBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    BufferedBatchIteratorMixin,
]
TrainIterator = make_iterator('TrainIterator', train_iterator_mixins)

test_iterator_mixins = [
    BufferedBatchIteratorMixin,
]
TestIterator = make_iterator('TestIterator', test_iterator_mixins)

train_iterator_kwargs = {
    'buffer_size': 5,
    'batch_size': batch_size,
    'flip_horizontal_p': 0.5,
    'flip_vertical_p': 0.5,
    'affine_p': 0.5,
    'affine_scale_choices': np.linspace(0.75, 1.25, 5),
    'affine_translation_choices': np.arange(-24, 24, 4),
    'affine_rotation_choices': np.arange(0, 360, 36),
}
train_iterator = TrainIterator(**train_iterator_kwargs)

test_iterator_kwargs = {
    'buffer_size': 5,
    'batch_size': batch_size,
}
test_iterator = TestIterator(**test_iterator_kwargs)

save_weights = SaveWeights('./model_weights.pkl', only_best=True, pickle=False)
save_training_history = SaveTrainingHistory('./model_history.pkl')
plot_training_history = PlotTrainingHistory('./training_history.png')

net = NeuralNet(
    layers=[
        ('in', InputLayer),

        ('l1c1', Conv2DDNNLayer),
        ('l1c2', Conv2DDNNLayer),
        ('l1p', MaxPool2DDNNLayer),

        ('l2c1', Conv2DDNNLayer),
        ('l2c2', Conv2DDNNLayer),
        ('l2p', MaxPool2DDNNLayer),

        ('l7', DenseLayer),
        ('l7drop', DropoutLayer),

        ('out', DenseLayer),
    ],

    in_shape=(None, 1, image_size, image_size),

    l1c1_num_filters=32, l1c1_filter_size=(3, 3), l1c1_border_mode='same',
    l1c2_num_filters=16, l1c2_filter_size=(3, 3), l1c2_border_mode='same',
    l1p_pool_size=(3, 3),
    l1p_stride=2,

    l2c1_num_filters=64, l2c1_filter_size=(3, 3), l2c1_border_mode='same',
    l2c2_num_filters=32, l2c2_filter_size=(3, 3), l2c2_border_mode='same',
    l2p_pool_size=(3, 3),
    l2p_stride=2,

    l7_num_units=64,
    l7drop_p=0.5,

    out_num_units=n_classes,
    out_nonlinearity=nonlinearities.softmax,

    regression=False,
    objective_loss_function=objectives.categorical_crossentropy,

    update=updates.rmsprop,
    update_learning_rate=1e-3,

    eval_size=0.1,
    batch_iterator_train=train_iterator,
    batch_iterator_test=test_iterator,

    on_epoch_finished=[
        save_weights,
        save_training_history,
        plot_training_history
    ],

    verbose=10,
    max_epochs=100
)

net.fit(X_train, y_train)

# Load the best weights from pickled model
net.load_params_from('./model_weights.pkl')

score = net.score(X_test, y_test)
print 'Final score %.4f' % score
