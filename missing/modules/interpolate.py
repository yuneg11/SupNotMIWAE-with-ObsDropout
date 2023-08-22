# Base code of RNN
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Base code of GRU-D
#
# MIT License
#
# Copyright (c) 2018 Zhengping Che
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


from collections import namedtuple

import tensorflow as tf

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Support auto-completion in IDEs.
    from keras.api._v2 import keras
    from keras.api._v2.keras import layers
else:
    from tensorflow import keras
    from tensorflow.keras import layers

from keras import backend
from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers

from keras.utils import tf_utils

from .gru_d import exp_relu, exp_softplus


__all__ = [
    "InterpolateInput",
    "InterpolateState",
    "DecayCell",
    "ForwardCell",
    "LinearScan",
    "DecayInterpolate",
    "LinearInterpolate",
]


CONSTANT_INIT = initializers.Constant(0.05)
InterpolateInput = namedtuple("InterpolateInput", ["values", "mask", "times"])
InterpolateState = namedtuple("InterpolateState", ["x_keep", "t_keep"])


class DecayCell(layers.AbstractRNNCell):
    def __init__(
        self,
        use_bias=True,
        decay="exp_softplus",
        decay_initializer="zeros",
        bias_initializer=CONSTANT_INIT,
        decay_regularizer=None,
        bias_regularizer=None,
        decay_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.use_bias = use_bias
        with keras.utils.custom_object_scope({"exp_relu": exp_relu, "exp_softplus": exp_softplus}):
            self.decay = None if decay is None else activations.get(decay)

        self.decay_initializer = initializers.get(decay_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.decay_regularizer = regularizers.get(decay_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.decay_constraint = constraints.get(decay_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self._input_dim = input_shape[0][-1]

        self.decay_kernel = self.add_weight(
            shape=(self._input_dim,),
            name="decay_kernel",
            initializer=self.decay_initializer,
            regularizer=self.decay_regularizer,
            constraint=self.decay_constraint,
        )
        if self.use_bias:
            self.decay_bias = self.add_weight(
                shape=(self._input_dim,),
                name="decay_bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

        self.built = True

    def call(self, inputs, states, training=None):
        x_input, m_input, t_input = inputs
        x_last, t_last = states
        t_delta = t_input - t_last

        gamma_di = t_delta * self.decay_kernel
        if self.use_bias:
            gamma_di = backend.bias_add(gamma_di, self.decay_bias)
        gamma_di = self.decay(gamma_di)

        x_t    = tf.where(m_input, x_input, gamma_di * x_last + (1 - gamma_di) * x_input)
        # x_t    = tf.where(m_input, x_input, gamma_di * x_last)
        x_keep = tf.where(m_input, x_input, x_last)
        t_keep = tf.where(m_input, t_input, t_last)
        return x_t, InterpolateState(x_keep, t_keep)

    @property
    def state_size(self):
        return InterpolateState(x_keep=self._input_dim, t_keep=self._input_dim)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is None:
            return InterpolateState(
                tf.zeros((batch_size, self._input_dim), dtype=dtype),
                tf.zeros((batch_size, self._input_dim), dtype=dtype),
            )
        else:
            return InterpolateState(
                tf.zeros((batch_size, self._input_dim), dtype=dtype),
                inputs.times[:, 0, :],
            )


class ForwardCell(keras.layers.AbstractRNNCell):
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self._input_dim = input_shape[0][-1]
        self.built = True

    def call(self, inputs, states, training=None):
        x_input, m_input, t_input = inputs
        x_last, t_last = states

        x_keep = tf.where(m_input, x_input, x_last)
        t_keep = tf.where(m_input, t_input, t_last)
        state = InterpolateState(x_keep, t_keep)
        return state, state

    @property
    def state_size(self):
        return InterpolateState(x_keep=self._input_dim, t_keep=self._input_dim)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is None:
            return InterpolateState(
                tf.zeros((batch_size, self._input_dim), dtype=dtype),
                tf.zeros((batch_size, self._input_dim), dtype=dtype),
            )
        else:
            return InterpolateState(
                tf.zeros((batch_size, self._input_dim), dtype=dtype),
                tf.reduce_max(inputs.times, axis=1) if self.go_backwards else inputs.times[:, 0, :],
            )


class LinearScan(layers.RNN):
    def __init__(
        self,
        unroll=False,
        go_backwards=False,
        **kwargs,
    ):

        cell = ForwardCell(
            dtype=kwargs.get("dtype"),
            trainable=kwargs.get("trainable", False),
        )

        super().__init__(
            cell,
            return_sequences=True,
            return_state=False,
            go_backwards=go_backwards,
            stateful=False,
            unroll=unroll,
            **kwargs,
        )


class DecayInterpolate(layers.RNN):
    def __init__(
        self,
        use_bias=True,
        decay="exp_relu",
        decay_initializer="zeros",
        bias_initializer=CONSTANT_INIT,
        decay_regularizer=None,
        bias_regularizer=None,
        decay_constraint=None,
        bias_constraint=None,
        unroll=False,
        **kwargs,
    ):

        cell = DecayCell(
            use_bias=use_bias,
            decay=decay,
            decay_initializer=decay_initializer,
            decay_regularizer=decay_regularizer,
            decay_constraint=decay_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            dtype=kwargs.get("dtype"),
            trainable=kwargs.get("trainable", True),
        )

        super().__init__(
            cell,
            return_sequences=True,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=unroll,
            **kwargs,
        )


class LinearInterpolate(keras.layers.Layer):
    def __init__(
        self,
        unroll=False,
        **kwargs,
    ):
        super().__init__()

        self.forward_scan  = LinearScan(unroll=unroll, go_backwards=False, **kwargs)
        self.backward_scan = LinearScan(unroll=unroll, go_backwards=True,  **kwargs)

    def call(self, inputs, mask=None, training=None):
        forwards  = self.forward_scan(inputs, mask=mask, training=training)
        backwards = self.backward_scan(inputs, mask=mask, training=training)

        x_t, t = inputs.values, inputs.times
        x_last, t_last = forwards.x_keep, forwards.t_keep
        x_next = tf.reverse(backwards.x_keep, axis=[1])
        t_next = tf.reverse(backwards.t_keep, axis=[1])

        # Linear interpolation. See https://en.wikipedia.org/wiki/Linear_interpolation
        x_itp = (x_last * (t_next - t) + x_next * (t - t_last)) / (t_next - t_last)
        x_itp = tf.where(tf.math.is_nan(x_itp), 0., x_itp)
        x_itp = tf.where(tf.math.is_inf(x_itp), 0., x_itp)

        return tf.where(inputs.mask, x_t, x_itp)
