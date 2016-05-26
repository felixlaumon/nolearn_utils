import numpy as np
from lasagne.utils import floatX


class DecayParameter(object):
    def __init__(self, param_name, start, stop, delay=0):
        self.param_name = param_name
        self.start = start
        self.stop = stop
        self.delay = delay

    def __call_(self, net, train_history):
        raise NotImplementedError()


class LinearDecayParameter(DecayParameter):
    """
    Linearly decay a parameter at each epoch
    Based on https://github.com/dnouri/kfkd-tutorial
    """
    def __call__(self, net, train_history):
        if not hasattr(self, 'param_values'):
            self.param_values = np.linspace(self.start, self.stop, net.max_epochs)

        epoch = train_history[-1]['epoch'] - self.delay
        if epoch > 0:
            new_value = floatX(self.param_values[epoch - 1])
            getattr(net, self.param_name).set_value(new_value)


class ExponentialDecayParameter(DecayParameter):
    """
    Expoentially decay a parameter at each epoch
    """
    pass


class ScheduleUpdateParameter(object):
    """
    Update a parameter at a pre-determined schedule
    """
    def __init__(self, param_name, schedule):
        self.param_name = param_name
        self.schedule = schedule


class DecayWhenNoImprovement(object):
    """
    Decay a parameter when the training loss does not improve
    """
    pass
