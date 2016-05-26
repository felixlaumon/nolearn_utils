import pytest
from mock import Mock
import numpy as np


def test_early_stopping():
    from nolearn_utils.handlers.training import EarlyStopping
    nn = Mock()
    early_stopping = EarlyStopping(
        metrics='valid_loss', patience=3, higher_is_better=False
    )

    train_history = [
        {'epoch': 1, 'valid_loss': 10},
        {'epoch': 2, 'valid_loss': 9},
        {'epoch': 3, 'valid_loss': 8},
        {'epoch': 4, 'valid_loss': 7},
        {'epoch': 5, 'valid_loss': 7},
        {'epoch': 6, 'valid_loss': 7},
        {'epoch': 7, 'valid_loss': 7}
    ]
    for i in range(1, 6):
        early_stopping(nn, train_history[:i])
    assert nn.get_all_params_values.call_count == 4

    with pytest.raises(StopIteration):
        early_stopping(nn, train_history)


def test_linear_decay_parameter():
    from nolearn_utils.handlers.update_param import LinearDecayParameter

    linear_decay = LinearDecayParameter(
        'update_learning_rate', start=1, stop=0
    )

    nn = Mock(max_epochs=11)
    train_history = [
        {'epoch': 1},
        {'epoch': 2},
        {'epoch': 3}
    ]

    for i in range(1, len(train_history) + 1):
        linear_decay(nn, train_history[:i])

    assert nn.update_learning_rate.set_value.call_count == 3
    actual_values = nn.update_learning_rate.set_value.call_args_list
    for actual_v, expected_v in zip(actual_values, [1.0, 0.9, 0.8]):
        assert np.isclose(actual_v[0][0], expected_v)


def test_linear_decay_parameter_with_delay():
    from nolearn_utils.handlers.update_param import LinearDecayParameter

    linear_decay = LinearDecayParameter(
        'update_learning_rate', start=1, stop=0, delay=5
    )

    nn = Mock(max_epochs=11)
    train_history = [
        {'epoch': 1},
        {'epoch': 2},
        {'epoch': 3},
        {'epoch': 4},
        {'epoch': 5},
        {'epoch': 6},
        {'epoch': 7},
        {'epoch': 8}
    ]

    for i in range(1, len(train_history) + 1):
        linear_decay(nn, train_history[:i])

    assert nn.update_learning_rate.set_value.call_count == 3
    actual_values = nn.update_learning_rate.set_value.call_args_list
    for actual_v, expected_v in zip(actual_values, [1.0, 0.9, 0.8]):
        assert np.isclose(actual_v[0][0], expected_v)
