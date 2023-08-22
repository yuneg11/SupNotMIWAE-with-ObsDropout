# Original implementation: https://github.com/PeterChe1990/GRU-D
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


from collections.abc import Sequence

import tensorflow as tf

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # This is a hack to make VS Code intellisense work
    # from tensorflow.python import keras
    from keras.api._v2 import keras
else:
    keras = tf.keras

from ..modules import GRUD, GRUDInput, GRUDState


class GRUDModel(keras.Model):
    def __init__(
        self,
        output_activation,
        output_dims,
        n_units: int = 32,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
    ):
        self._config = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        super().__init__()

        if isinstance(output_dims, Sequence):
            # We have an online prediction scenario
            assert output_dims[0] is None
            self.return_sequences = True
            output_dims = output_dims[1]
        else:
            self.return_sequences = False
        self.output_dims = output_dims
        self.n_units = n_units
        self.rnn = GRUD(
            n_units, dropout=dropout, recurrent_dropout=recurrent_dropout,
            return_sequences=self.return_sequences,
        )
        self.output_layer = keras.layers.Dense(output_dims, activation=output_activation)

    def build(self, input_shape):
        demo, times, values, measurements, lengths = input_shape
        self.rnn.build(GRUDInput(values=values, mask=measurements, times=times + (1,)))
        self.demo_encoder = keras.Sequential([
            keras.layers.Dense(self.n_units, activation="relu"),
            keras.layers.Dense(self.rnn.cell.state_size[0])
        ], name="demo_encoder")
        self.demo_encoder.build(demo)

    def get_config(self):
        return self._config

    def call(self, inputs):
        demo, times, values, measurements, lengths = inputs

        if len(lengths.get_shape()) == 2:
            lengths = tf.squeeze(lengths, -1)

        times = tf.expand_dims(times, -1)

        demo_encoded = self.demo_encoder(demo)
        initial_state = GRUDState(
            demo_encoded,
            tf.zeros(tf.stack([tf.shape(demo)[0], self.rnn.cell._input_dim])),
            tf.tile(times[:, 0, :], [1, self.rnn.cell._input_dim])
        )

        mask = tf.sequence_mask(lengths, maxlen=tf.shape(times)[1])
        grud_output = self.rnn(
            GRUDInput(
                values=values,
                mask=measurements,
                times=times
            ),
            mask=mask,
            initial_state=initial_state
        )
        output = self.output_layer(grud_output)

        return output

    def data_preprocessing_fn(self):
        def one_hot_label(ts, labels):      # batch 축 존재 x
            # Ignore demographics for now
            demo, X, Y, measurements, lengths = ts 
            # labels = [50] -> [50,11]
            if self.return_sequences and self.output_dims==11:
                depth = 11
                labels = tf.one_hot(labels,depth)
            else:
                pass
            return (demo, X, Y, measurements, lengths), labels

        return one_hot_label