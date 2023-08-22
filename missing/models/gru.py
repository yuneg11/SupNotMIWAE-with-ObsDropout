# Original implementation: https://github.com/BorgwardtLab/Set_Functions_for_Time_Series
#
# Copyright 2020 Max Horn
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AD CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.N


from collections.abc import Sequence

import tensorflow as tf

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # This is a hack to make VS Code intellisense work
    # from tensorflow.python import keras
    from keras.api._v2 import keras
else:
    keras = tf.keras


__all__ = [
    "GRUModel",
]


class GRUModel(keras.Model):
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

        self.n_units = n_units
        self.rnn = keras.layers.GRU(
            n_units, dropout=dropout, recurrent_dropout=recurrent_dropout,
            return_sequences=self.return_sequences,
        )
        self.output_layer = keras.layers.Dense(output_dims, activation=output_activation)

    def get_config(self):
        return self._config

    def call(self, inputs):
        demo, times, values, measurements, lengths = inputs

        if len(lengths.get_shape()) == 2:
            lengths = tf.squeeze(lengths, -1)

        mask = tf.sequence_mask(lengths, maxlen=tf.shape(times)[1])

        gru_output = self.rnn(values, mask=mask)
        output = self.output_layer(gru_output)

        return output
