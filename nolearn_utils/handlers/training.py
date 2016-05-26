import numpy as np
from datetime import datetime
import cPickle as pickle


class EarlyStopping(object):
    """
    Stop training if the specified metrics stop improving for `patience` number
    of epoch

    Based on https://github.com/dnouri/kfkd-tutorial

    Parameters
    ----------
    metrics : string, default='valid_loss'
        Name of the metrics / scorer to be monitor
    patience : integer, default=50
        Number of epoch
    verbose : boolean, default=False
        TODO
    higher_is_better: boolean, default=False
        TODO
    """
    def __init__(self, metrics='valid_loss', patience=50, verbose=False, higher_is_better=False):
        self.patience = patience
        if higher_is_better:
            self.best_metrics = 0
        else:
            self.best_metrics = np.inf

        self.metrics = metrics
        self.best_metrics_epoch = 0
        self.best_weights = None
        self.verbose = verbose
        self.higher_is_better = higher_is_better

    def __call__(self, nn, train_history):
        current_metrics = train_history[-1][self.metrics]
        current_epoch = train_history[-1]['epoch']

        if self.higher_is_better:
            has_metrics_improved = current_metrics > self.best_metrics
        else:
            has_metrics_improved = current_metrics < self.best_metrics

        if has_metrics_improved:
            if self.verbose:
                print 'Best metrics at epoch %i' % current_epoch

            self.best_metrics = current_metrics
            self.best_metrics_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()

        elif self.best_metrics_epoch + self.patience <= current_epoch:
            print 'Early stopping.'
            print 'Best metrics was %.6f at epoch %d' % (
                self.best_metrics, self.best_metrics_epoch
            )
            nn.load_params_from(self.best_weights)
            raise StopIteration()

        else:
            if self.verbose:
                print 'Not updating'
                print current_metrics, self.best_metrics
