import numpy as np
import lasagne as nn

from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import softmax

from sklearn.metrics import log_loss

from nolearn.lasagne import NeuralNet
from nolearn.lasagne.handlers import SaveWeights

from nolearn_utils.iterators import BatchIterator

from nolearn_utils.transformers import (
    RandomHorizontalFlipTransformer,
    RandomAffineTransformer
)

from nolearn_utils.handlers import (
    SaveTrainingHistory,
    PlotTrainingHistory,
    EarlyStopping
)


from nolearn_utils.layer_marco import (
    vgg
)


image_size = 28
batch_size = 1024
n_classes = 10


model_weights_fname = 'examples/mnist/weights.pkl'
model_history_fname = 'examples/mnist/history.csv'
model_graph_fname = 'examples/mnist/history.png'


def get_conv_kwargs():
    leaky_alpha = 1 / 3.0
    glorot_gain = np.sqrt(2 / (1 + leaky_alpha ** 2))
    nonlinearity = nn.nonlinearities.LeakyRectify(leaky_alpha)
    W = nn.init.GlorotNormal(glorot_gain)

    return dict(
        W=W, nonlinearity=nonlinearity, pad='same'
    )


def get_layers():
    l = InputLayer(name='input', shape=(None, 1, image_size, image_size))
    # 28x28

    l = vgg(
        l, name='1', num_layers=2,
        num_filters=16, filter_size=3, downsample='stride',
        drop_p=0.0, bn=True,
        **get_conv_kwargs()
    )
    # 14x14

    l = vgg(
        l, name='2', num_layers=2,
        num_filters=32, filter_size=3, downsample='stride',
        drop_p=0.0, bn=True,
        **get_conv_kwargs()
    )
    # 7x7

    l = vgg(
        l, name='3', num_layers=3,
        num_filters=64, filter_size=3, downsample='stride',
        drop_p=0.0, bn=True,
        **get_conv_kwargs()
    )

    l = nn.layers.GlobalPoolLayer(l, name='gp')
    l = nn.layers.DropoutLayer(l, name='gpdrop', p=0.5)

    l = DenseLayer(l, name='out', num_units=n_classes, nonlinearity=softmax)
    return l


def get_iterators():
    train_iterator = BatchIterator(
        batch_size=batch_size,
        worker_batch_size=batch_size // 2, n_workers=8,
        shuffle=True
    )

    train_iterator.add_transformer(
        RandomHorizontalFlipTransformer(p=0.5)
    )

    train_iterator.add_transformer(RandomAffineTransformer(
        p=1.0,
        scale=np.arange(-0.9, 1.1, 0.05),
        rotation=np.arange(-5, 6, 1),
        shear=np.arange(-0.1, 0.11, 0.01),
        translation_y=np.arange(-3, 4, 1),
        translation_x=np.arange(-3, 4, 1)
    ))

    test_iterator = BatchIterator(
        batch_size=batch_size,
        n_workers=1,
        shuffle=False
    )

    return train_iterator, test_iterator


def get_net(layers, train_iterator, test_iterator):
    save_weights = SaveWeights(
        model_weights_fname, only_best=True, pickle=False, verbose=True
    )
    save_training_history = SaveTrainingHistory(
        model_history_fname, output_format='csv'
    )
    plot_training_history = PlotTrainingHistory(model_graph_fname)
    early_stopping = EarlyStopping(patience=100)

    net = NeuralNet(
        layers=layers,

        regression=False,

        objective_loss_function=nn.objectives.categorical_crossentropy,
        # objective_l2=1e-6,

        update=nn.updates.adam,
        update_learning_rate=1e-4,

        batch_iterator_train=train_iterator,
        batch_iterator_test=test_iterator,

        on_epoch_finished=[
            save_weights,
            save_training_history,
            plot_training_history,
            early_stopping,
        ],

        verbose=10,

        custom_scores=[
            ('logloss', log_loss)
        ],

        max_epochs=2000,
    )

    return net
