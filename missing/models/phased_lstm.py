# Phased LSTM implementation based on the version in tensorflow contrib.

# See: https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1915-L2064

# Due to restructurings in tensorflow some adaptions were required. This
# implementation does not use global naming of variables and thus is compatible
# with the new keras style paradime.


from collections.abc import Sequence
from collections import namedtuple

import tensorflow as tf

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # This is a hack to make VS Code intellisense work
    from keras.api._v2 import keras
else:
    keras = tf.keras

from .. import functional as F


PhasedLSTMInput = namedtuple("PhasedLSTMInput", ["times", "x"])
LSTMStateTuple = namedtuple("LSTMStateTuple", ("c", "h"))


def _random_exp_initializer(minval, maxval, seed=None, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        del partition_info  # Unused.
        return tf.math.exp(tf.random.uniform(shape, tf.math.log(minval), tf.math.log(maxval), dtype, seed=seed))
    return _initializer


def _random_exp_uniform_initializer(minval, maxval, seed=None, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        del partition_info  # Unused.
        return tf.random.uniform(
            shape, 0.,
            tf.math.exp(tf.random.uniform(shape, tf.math.log(minval), tf.math.log(maxval), dtype, seed=seed)),
            dtype, seed=seed
        )
    return _initializer


class PhasedLSTMCell(tf.keras.layers.Layer):
    """Phased LSTM recurrent network cell.

    https://arxiv.org/pdf/1610.09513v1.pdf
    """

    def __init__(
        self,
        units,
        use_peepholes=False,
        leak=0.001,
        ratio_on=0.1,
        trainable_ratio_on=True,
        period_init_min=0.5,
        period_init_max=1000.0,
    ):
        super().__init__()

        self.units = units
        self._use_peepholes = use_peepholes
        self._leak = leak
        self._ratio_on = ratio_on
        self._trainable_ratio_on = trainable_ratio_on
        self._period_init_min = period_init_min
        self._period_init_max = period_init_max

        self.period = self.add_weight(
            "period", shape=(self.units,),
            initializer=_random_exp_initializer(self._period_init_min, self._period_init_max),
        )
        self.phase = self.add_weight(
            "phase", shape=(self.units,),
            initializer=_random_exp_uniform_initializer(self._period_init_min, self._period_init_max),
        )
        self.ratio_on = self.add_weight(
            "ratio_on", shape=(self.units,),
            initializer=keras.initializers.Constant(self._ratio_on),
            trainable=self._trainable_ratio_on,
        )

        self.linear1 = keras.layers.Dense(2 * self.units, use_bias=True, activation="sigmoid", name="linear1")
        self.linear2 = keras.layers.Dense(self.units,     use_bias=True, activation="tanh",    name="linear2")
        self.linear3 = keras.layers.Dense(self.units,     use_bias=True, activation="sigmoid", name="linear3")

    @property
    def state_size(self):
        return LSTMStateTuple(self.units, self.units)

    @property
    def output_size(self):
        return self.units

    def _mod(self, x, y):
        return tf.stop_gradient(tf.math.mod(x, y) - x) + x

    def _get_cycle_ratio(self, time):
        phase = tf.cast(self.phase, dtype=time.dtype)
        period = tf.cast(self.period, dtype=time.dtype)
        shifted_time = time - phase
        cycle_ratio = self._mod(shifted_time, period) / period
        return tf.cast(cycle_ratio, dtype=tf.float32)

    def call(self, inputs, state):
        (c_prev, h_prev) = state
        time, x = inputs.times, inputs.x

        if self._use_peepholes:
            input_mask_and_output_gate = tf.concat([x, h_prev, c_prev], axis=-1)
        else:
            input_mask_and_output_gate = tf.concat([x, h_prev], axis=-1)

        mask_gates = self.linear1(input_mask_and_output_gate)
        input_gate, forget_gate = tf.split(mask_gates, axis=1, num_or_size_splits=2)
        new_input = self.linear2(tf.concat([x, h_prev], axis=-1))
        new_c = (c_prev * forget_gate + input_gate * new_input)
        output_gate = self.linear3(input_mask_and_output_gate)
        new_h = tf.tanh(new_c) * output_gate

        cycle_ratio = self._get_cycle_ratio(time)
        k_up = 2 * cycle_ratio / self.ratio_on
        k_down = 2 - k_up
        k_closed = self._leak * cycle_ratio

        k = tf.where(cycle_ratio < self.ratio_on, k_down, k_closed)
        k = tf.where(cycle_ratio < 0.5 * self.ratio_on, k_up, k)

        new_c = k * new_c + (1 - k) * c_prev
        new_h = k * new_h + (1 - k) * h_prev

        new_state = LSTMStateTuple(new_c, new_h)
        return new_h, new_state


class PhasedLSTMModel(tf.keras.Model):
    def __init__(
        self,
        output_activation,
        output_dims,
        n_units,
        use_peepholes,
        leak,
        period_init_max,
    ):
        self._config = {name: val for name, val in locals().items() if name not in ["self", "__class__"]}
        super().__init__()

        if isinstance(output_dims, Sequence):
            # We have an online prediction scenario
            assert output_dims[0] is None
            self.return_sequences = True
            output_dims = output_dims[1]
        else:
            self.return_sequences = False

        self.demo_encoder = keras.Sequential([
            keras.layers.Dense(n_units, activation="relu"),
            keras.layers.Dense(n_units * 2)
        ], name="demo_encoder")

        self.rnn = keras.layers.RNN(
            PhasedLSTMCell(
                n_units, use_peepholes=use_peepholes,
                leak=leak, period_init_max=period_init_max
            ),
            return_sequences=self.return_sequences,
        )
        self.output_layer = keras.layers.Dense(output_dims, activation=output_activation)

    def call(self, inputs):
        demo, times, values, measurements, dt, lengths = inputs

        demo_encoding = self.demo_encoder(demo)
        initial_state = LSTMStateTuple(*tf.split(demo_encoding, 2, axis=-1))
        values = tf.concat((values, tf.cast(measurements, tf.float32), dt), axis=-1)
        mask = tf.sequence_mask(lengths, maxlen=tf.shape(times)[1])

        lstm_output = self.rnn(PhasedLSTMInput(times=times, x=values), mask=mask, initial_state=initial_state)
        output = self.output_layer(lstm_output)

        return output

    def data_preprocessing_fn(self):
        def add_delta_t_tensor(inputs, label):
            demo, times, values, measurements, length = inputs
            times = tf.expand_dims(times, -1)
            dt = F.delta_t(times, values, measurements)
            if self.return_sequences:
                depth = 11
                label = tf.one_hot(label,depth)

            return (demo, times, values, measurements, dt, length), label
        return add_delta_t_tensor

    def get_config(self):
        return self._config
