import cPickle as pickle
import pandas as pd
import numpy as np


class SaveTrainingHistory(object):
    """
    Save net.training_history as pickle at each epoch
    """
    def __init__(self, path, verbose=0, output_format='csv'):
        assert output_format in ['pickle', 'csv']
        self.path = path
        self.verbose = verbose
        self.output_format = output_format

    def __call__(self, nn, train_history):
        if self.output_format == 'pickle':
            with open(self.path, 'wb') as f:
                pickle.dump(train_history, f, -1)
        else:
            df = pd.DataFrame(train_history)
            df.to_csv(self.path, index=False)


class PlotTrainingHistory(object):
    """
    Plot the training history as a line chart at each epoch and render to a
    .png file using the matplotlib `Agg` backend

    The png contains 2 graphs:
    - Training and validation loss over time
    - Accruacy over time
    """
    def __init__(self, path, log_scale=False, figsize=(20, 8)):
        # TODO warn the user if not using the Agg backend
        self.path = path
        self.log_scale = log_scale
        self.figsize = figsize

    def __call__(self, nn, train_history):
        import matplotlib.pyplot as plt

        valid_accuracy = np.asarray([history['valid_accuracy'] for history in train_history])
        train_loss = np.asarray([history['train_loss'] for history in train_history])
        valid_loss = np.asarray([history['valid_loss'] for history in train_history])

        plt.figure(figsize=self.figsize)

        plt.subplot(1, 2, 1)
        plt.title('Loss over time')
        plt.plot(train_loss, label='Training loss')
        plt.plot(valid_loss, label='Validation loss')
        if self.log_scale is True:
            plt.yscale('log')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title('Accuracy against training epoch')
        plt.plot(valid_accuracy, label='Validation accuracy')
        plt.legend()

        plt.savefig(self.path)
        plt.close()


class PastaLog(object):
    """
    Send training loss, validation loss and validation accuracy to pastalog
    """
    def __init__(self, model_name, url='http://localhost:8120'):
        from pastalog import Log
        self.model_name = model_name
        self.url = url
        self.log = Log(self.url, self.model_name)

    def __call__(self, nn, train_history):
        hist = train_history[-1]
        epoch = len(train_history)
        train_loss = hist['train_loss']
        valid_loss = hist['valid_loss']
        valid_accuracy = hist['valid_accuracy']

        try:
            # Seems json cannot encode float32?
            self.log.post('train_loss', value=float(train_loss), step=epoch)
            self.log.post('valid_loss', value=float(valid_loss), step=epoch)
            self.log.post('valid_accuracy', value=float(valid_accuracy), step=epoch)
        except Exception, e:
            # TODO catch only those raised by request
            print 'PastaLog failed', e
            pass
