from .gru_d import GRUD


__all__ = [
    "GRUForward",
]


class GRUForward(GRUD):
    def __init__(
        self,
        units,
        x_imputation="forward",
        input_decay=None,
        hidden_decay=None,
        use_decay_bias=False,
        feed_masking=False,
        masking_decay=None,
        decay_initializer="zeros",
        decay_regularizer=None,
        decay_constraint=None,
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.,
        recurrent_dropout=0.,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        reset_after=False,
        **kwargs,
    ):
        super().__init__(
            units,
            x_imputation=x_imputation,
            input_decay=input_decay,
            hidden_decay=hidden_decay,
            use_decay_bias=use_decay_bias,
            feed_masking=feed_masking,
            masking_decay=masking_decay,
            decay_initializer=decay_initializer,
            decay_regularizer=decay_regularizer,
            decay_constraint=decay_constraint,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            reset_after=reset_after,
            **kwargs,
        )