from collections import namedtuple

import tensorflow as tf

from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers

from keras.engine import base_layer
from keras.engine.input_spec import InputSpec

from keras.layers.rnn import gru_lstm_utils, rnn_utils
from keras.layers.rnn.gru import GRUCell
from keras.layers.rnn.base_rnn import RNN
from keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin

from keras.utils import tf_utils

from tensorflow.python.platform import tf_logging as logging

import tensorflow_probability as tfp

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Support auto-completion in IDEs.
    from tensorflow_probability.python.distributions import Normal
else:
    Normal = tfp.distributions.Normal


RECURRENT_DROPOUT_WARNING_MSG = (
    "RNN `implementation=2` is not supported when `recurrent_dropout` is set. "
    "Using `implementation=1`."
)


__all__ = [
    "GRUAutoCell",
    "GRUAuto",
]


# See [RNN with Keras Guide](https://www.tensorflow.org/guide/keras/rnn#rnns_with_listdict_inputs_or_nested_inputs)
# for more details.


class GRUAutoCell(DropoutRNNCellMixin, base_layer.BaseRandomLayer):
    def __init__(
        self,
        hidden_units,
        output_units,
        activation="tanh",
        recurrent_activation="sigmoid",
        mu_activation=None,
        sigma_activation="softplus",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        auto_initializer="glorot_uniform",
        dist_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        auto_regularizer=None,
        dist_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        auto_constraint=None,
        dist_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        reset_after=True,
        **kwargs,
    ):
        if hidden_units < 0:
            raise ValueError(
                f"Received an invalid value for argument `hidden_units`, "
                f"expected a positive integer, got {hidden_units}."
            )

        if output_units < 0:
            raise ValueError(
                f"Received an invalid value for argument `output_units`, "
                f"expected a positive integer, got {output_units}."
            )

        # By default use cached variable under v2 mode, see b/143699808.
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop("enable_caching_device", True)
        else:
            self._enable_caching_device = kwargs.pop("enable_caching_device", False)

        super().__init__(**kwargs)

        self.hidden_units = hidden_units
        self.output_units = output_units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.mu_activation = activations.get(mu_activation)
        self.sigma_activation = activations.get(sigma_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.auto_initializer = initializers.get(auto_initializer)
        self.dist_initializer = initializers.get(dist_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.auto_regularizer = regularizers.get(auto_regularizer)
        self.dist_regularizer = regularizers.get(dist_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.auto_constraint = constraints.get(auto_constraint)
        self.dist_constraint = constraints.get(dist_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))

        implementation = kwargs.pop("implementation", 2)

        if self.recurrent_dropout != 0 and implementation != 1:
            logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
            self.implementation = 1
        else:
            self.implementation = implementation

        self.reset_after = reset_after
        self.state_size = [tf.TensorShape([self.hidden_units]), tf.TensorShape([self.output_units])]  # (h, mu)
        self.output_size = [tf.TensorShape([self.output_units]) for _ in range(3)]  # (y, mu, sigma)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]
        default_caching_device = rnn_utils.caching_device(self)
        self.kernel = self.add_weight(
            shape=(input_dim, self.hidden_units * 3),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.hidden_units, self.hidden_units * 3),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device,
        )
        self.auto_kernel = self.add_weight(
            shape=(self.output_units, self.hidden_units * 3),
            name="auto_kernel",
            initializer=self.auto_initializer,
            regularizer=self.auto_regularizer,
            constraint=self.auto_constraint,
            caching_device=default_caching_device,
        )
        self.mu_kernel = self.add_weight(
            shape=(self.hidden_units, self.output_units),
            name="mu_kernel",
            initializer=self.dist_initializer,
            regularizer=self.dist_regularizer,
            constraint=self.dist_constraint,
            caching_device=default_caching_device,
        )
        self.sigma_kernel = self.add_weight(
            shape=(self.hidden_units, self.output_units),
            name="sigma_kernel",
            initializer=self.dist_initializer,
            regularizer=self.dist_regularizer,
            constraint=self.dist_constraint,
            caching_device=default_caching_device,
        )

        if self.use_bias:
            if not self.reset_after:
                bias_shape = (4, 3 * self.hidden_units,)
                # TODO: implement reset_after=False
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU
                # biases `(5 * 3 * self.hidden_units,)`, so that we can distinguish the
                # classes when loading and converting saved weights.
                bias_shape = (3, 3 * self.hidden_units)
                dist_bias_shape = (2, self.output_units)

            self.bias = self.add_weight(
                shape=bias_shape,
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device,
            )
            self.dist_bias = self.add_weight(
                shape=dist_bias_shape,
                name="dist_bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device,
            )
        else:
            self.bias = None
            self.dist_bias = None

        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1, y_tm1 = states

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=6)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=3)

        if self.use_bias:
            if not self.reset_after:
                input_bias, auto_bias = tf.unstack(self.bias)
                mu_bias, sigma_bias = tf.unstack(self.dist_bias)
                recurrent_bias = None
            else:
                input_bias, auto_bias, recurrent_bias = tf.unstack(self.bias)
                mu_bias, sigma_bias = tf.unstack(self.dist_bias)

        if self.implementation == 1:
            if 0.0 < self.dropout < 1.0:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
                y_tm1_z = y_tm1 * dp_mask[3]
                y_tm1_r = y_tm1 * dp_mask[4]
                y_tm1_h = y_tm1 * dp_mask[5]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs
                y_tm1_z = y_tm1
                y_tm1_r = y_tm1
                y_tm1_h = y_tm1

            x_z = backend.dot(inputs_z, self.kernel[:, : self.hidden_units])
            x_r = backend.dot(inputs_r, self.kernel[:, self.hidden_units : self.hidden_units * 2])
            x_h = backend.dot(inputs_h, self.kernel[:, self.hidden_units * 2 :])
            y_tm1_z = backend.dot(y_tm1_z, self.auto_kernel[:, : self.hidden_units])
            y_tm1_r = backend.dot(y_tm1_r, self.auto_kernel[:, self.hidden_units : self.hidden_units * 2])
            y_tm1_h = backend.dot(y_tm1_h, self.auto_kernel[:, self.hidden_units * 2 :])

            if self.use_bias:
                x_z = backend.bias_add(x_z, input_bias[: self.hidden_units])
                x_r = backend.bias_add(x_r, input_bias[self.hidden_units : self.hidden_units * 2])
                x_h = backend.bias_add(x_h, input_bias[self.hidden_units * 2 :])
                y_tm1_z = backend.bias_add(y_tm1_z, auto_bias[: self.hidden_units])
                y_tm1_r = backend.bias_add(y_tm1_r, auto_bias[self.hidden_units : self.hidden_units * 2])
                y_tm1_h = backend.bias_add(y_tm1_h, auto_bias[self.hidden_units * 2 :])

            if 0.0 < self.recurrent_dropout < 1.0:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            recurrent_z = backend.dot(h_tm1_z, self.recurrent_kernel[:, : self.hidden_units])
            recurrent_r = backend.dot(h_tm1_r, self.recurrent_kernel[:, self.hidden_units : self.hidden_units * 2])
            if self.reset_after and self.use_bias:
                recurrent_z = backend.bias_add(recurrent_z, recurrent_bias[: self.hidden_units])
                recurrent_r = backend.bias_add(recurrent_r, recurrent_bias[self.hidden_units : self.hidden_units * 2])

            z = self.recurrent_activation(x_z + y_tm1_z + recurrent_z)
            r = self.recurrent_activation(x_r + y_tm1_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = backend.dot(h_tm1_h, self.recurrent_kernel[:, self.hidden_units * 2 :])
                if self.use_bias:
                    recurrent_h = backend.bias_add(recurrent_h, recurrent_bias[self.hidden_units * 2 :])
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = backend.dot(r * h_tm1_h, self.recurrent_kernel[:, self.hidden_units * 2 :])

            hh = self.activation(x_h + y_tm1_h + recurrent_h)
        else:
            if 0.0 < self.dropout < 1.0:
                inputs = inputs * dp_mask[0]
                y_tm1 = y_tm1 * dp_mask[3]

            # inputs projected by all gate matrices at once
            matrix_x = backend.dot(inputs, self.kernel)
            matrix_y_tm1 = backend.dot(y_tm1, self.auto_kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = backend.bias_add(matrix_x, input_bias)
                matrix_y_tm1 = backend.bias_add(matrix_y_tm1, auto_bias)

            x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)
            y_tm1_z, y_tm1_r, y_tm1_h = tf.split(matrix_y_tm1, 3, axis=-1)

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = backend.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = backend.bias_add(matrix_inner, recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = backend.dot(h_tm1, self.recurrent_kernel[:, : 2 * self.hidden_units])

            recurrent_z, recurrent_r, recurrent_h = tf.split(matrix_inner, [self.hidden_units, self.hidden_units, -1], axis=-1)

            z = self.recurrent_activation(x_z + y_tm1_z + recurrent_z)
            r = self.recurrent_activation(x_r + y_tm1_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = backend.dot(r * h_tm1, self.recurrent_kernel[:, 2 * self.hidden_units :])

            hh = self.activation(x_h + y_tm1_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh

        # mu / sigma
        mu = backend.dot(h, self.mu_kernel)
        sigma = backend.dot(h, self.sigma_kernel)

        if self.use_bias:
            mu = backend.bias_add(mu, mu_bias)
            sigma = backend.bias_add(sigma, sigma_bias)

        mu = self.mu_activation(mu)
        sigma = self.sigma_activation(sigma)

        dist = Normal(loc=mu, scale=sigma)
        y = dist.sample()

        return (y, mu, sigma), ((h, y) if training else (h, mu))

    def get_config(self):
        config = {
            "units": self.hidden_units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(self.recurrent_activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "implementation": self.implementation,
            "reset_after": self.reset_after,
        }
        config.update(rnn_utils.config_for_enable_caching_device(self))
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    #     return rnn_utils.generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)


class GRUAuto(DropoutRNNCellMixin, RNN, base_layer.BaseRandomLayer):

    def __init__(
        self,
        hidden_units,
        output_units,
        activation="tanh",
        recurrent_activation="sigmoid",
        mu_activation=None,
        sigma_activation="softplus",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        auto_initializer="glorot_uniform",
        dist_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        auto_regularizer=None,
        dist_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        auto_constraint=None,
        dist_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        time_major=False,
        reset_after=True,
        **kwargs,
    ):

        if "enable_caching_device" in kwargs:
            cell_kwargs = {
                "enable_caching_device": kwargs.pop("enable_caching_device"),
            }
        else:
            cell_kwargs = {}

        if stateful:
            raise NotImplementedError("Stateful GRUAuto is not supported yet.")

        cell = GRUAutoCell(
            hidden_units,
            output_units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            mu_activation=mu_activation,
            sigma_activation=sigma_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            auto_initializer=auto_initializer,
            dist_initializer=dist_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            auto_regularizer=auto_regularizer,
            dist_regularizer=dist_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            auto_constraint=auto_constraint,
            dist_constraint=dist_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            reset_after=reset_after,
            dtype=kwargs.get("dtype"),
            trainable=kwargs.get("trainable", True),
            **cell_kwargs,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            time_major=time_major,
            **kwargs,
        )
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]
        self.output_size = self.cell.output_size

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # The input should be dense, padded with zeros. If a ragged input is fed
        # into the layer, it is padded and the row lengths are used for masking.
        inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
        is_ragged_input = row_lengths is not None
        self._validate_args_if_ragged(is_ragged_input, mask)

        # GRU does not support constants. Ignore it during process.
        inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)

        if isinstance(mask, list):
            mask = mask[0]

        input_shape = backend.int_shape(inputs)
        timesteps = input_shape[0] if self.time_major else input_shape[1]

        kwargs = {"training": training}
        self._maybe_reset_cell_dropout_mask(self.cell)

        def step(cell_inputs, cell_states):
            return self.cell(cell_inputs, cell_states, **kwargs)

        last_output, outputs, states = backend.rnn(
            step,
            inputs,
            initial_state,
            constants=None,
            go_backwards=self.go_backwards,
            mask=mask,
            unroll=self.unroll,
            input_length=row_lengths
            if row_lengths is not None
            else timesteps,
            time_major=self.time_major,
            zero_output_for_mask=self.zero_output_for_mask,
            return_all_outputs=self.return_sequences,
        )

        if self.return_sequences:
            output = backend.maybe_convert_to_ragged(
                is_ragged_input,
                outputs,
                row_lengths,
                go_backwards=self.go_backwards,
            )
        else:
            output = last_output

        if self.return_state:
            return [output] + list(states)
        else:
            return output

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def reset_after(self):
        return self.cell.reset_after

    def get_config(self):
        config = {
            "units": self.hidden_units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(self.recurrent_activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "reset_after": self.reset_after,
        }
        base_config = super().get_config()
        del base_config["cell"]
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
