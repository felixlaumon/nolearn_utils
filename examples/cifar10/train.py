import os
import numpy as np
import pandas as pd

from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, FeaturePoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates

from nolearn.lasagne import NeuralNet
from nolearn.lasagne.handlers import SaveWeights, PrintLayerInfo
from nolearn.lasagne.util import get_conv_infos

from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from nolearn_utils.iterators import (
    BufferedBatchIteratorMixin,
    ReadImageBatchIteratorMixin,
    RandomCropBatchIteratorMixin,
    ShuffleBatchIteratorMixin,
    LCNBatchIteratorMixin,
    RandomFlipBatchIteratorMixin,
    EqualizeAdaptHistBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    LCNBatchIteratorMixin,
    make_iterator
)
from nolearn_utils.hooks import SaveTrainingHistory, PlotTrainingHistory


def stratified_train_test_split(X, y, test_size=0.25, random_state=None):
    n_folds = int(1 / test_size)
    skf = StratifiedKFold(y, n_folds=n_folds, random_state=random_state)
    train_idx, test_idx = iter(skf).next()
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def load_data(test_size=0.25, random_state=None, data_dir='./data'):
    csv_fname = os.path.join(data_dir, 'trainLabels.csv')
    df = pd.read_csv(csv_fname)
    X = df['id'].apply(lambda i: '%s.png' % i).values
    y = LabelEncoder().fit_transform(df['label'].values)

    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=test_size, random_state=random_state)
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
    return X_train, X_test, y_train, y_test

# X_train, X_test are image file names
# They will be read in the iterator
X_train, X_test, y_train, y_test = load_data()

batch_size = 512
n_classes = 10
image_size = 32

train_iterator_mixins = [
    ShuffleBatchIteratorMixin,
    ReadImageBatchIteratorMixin,
    RandomFlipBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    BufferedBatchIteratorMixin,
]
TrainIterator = make_iterator('TrainIterator', train_iterator_mixins)

test_iterator_mixins = [
    ReadImageBatchIteratorMixin,
    BufferedBatchIteratorMixin,
]
TestIterator = make_iterator('TestIterator', test_iterator_mixins)

train_iterator_kwargs = {
    'buffer_size': 5,
    'batch_size': batch_size,
    'read_image_size': (image_size, image_size),
    'read_image_as_gray': False,
    'read_image_prefix_path': './data/train/',
    'flip_horizontal_p': 0.5,
    'flip_vertical_p': 0,
    'affine_p': 0.5,
    'affine_scale_choices': np.linspace(0.75, 1.25, 5),
    'affine_translation_choices': np.arange(-3, 4, 1),
    'affine_rotation_choices': np.arange(-45, 50, 5)
}
train_iterator = TrainIterator(**train_iterator_kwargs)

test_iterator_kwargs = {
    'buffer_size': 5,
    'batch_size': batch_size,
    'read_image_size': (image_size, image_size),
    'read_image_as_gray': False,
    'read_image_prefix_path': './data/train/',
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
        ('l1c3', Conv2DDNNLayer),
        ('l1p', MaxPool2DDNNLayer),

        ('l2c1', Conv2DDNNLayer),
        ('l2c2', Conv2DDNNLayer),
        ('l2c3', Conv2DDNNLayer),
        ('l2p', MaxPool2DDNNLayer),

        ('l6', DenseLayer),
        ('l6p', FeaturePoolLayer),
        ('l6drop', DropoutLayer),

        ('l7', DenseLayer),
        ('l7p', FeaturePoolLayer),
        ('l7drop', DropoutLayer),

        ('out', DenseLayer),
    ],

    in_shape=(None, 3, image_size, image_size),

    l1c1_num_filters=32, l1c1_filter_size=(3, 3), l1c1_border_mode='same',
    l1c2_num_filters=32, l1c2_filter_size=(3, 3), l1c2_border_mode='same',
    l1c3_num_filters=16, l1c3_filter_size=(3, 3), l1c3_border_mode='same',
    l1p_pool_size=(3, 3),
    l1p_stride=2,

    l2c1_num_filters=64, l2c1_filter_size=(3, 3), l2c1_border_mode='same',
    l2c2_num_filters=64, l2c2_filter_size=(3, 3), l2c2_border_mode='same',
    l2c3_num_filters=32, l2c3_filter_size=(3, 3), l2c3_border_mode='same',
    l2p_pool_size=(3, 3),
    l2p_stride=2,

    l6_num_units=256,
    l6p_pool_size=2,
    l6drop_p=0.5,

    l7_num_units=256,
    l7p_pool_size=2,
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
