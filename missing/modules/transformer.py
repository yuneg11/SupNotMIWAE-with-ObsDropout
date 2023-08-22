from collections.abc import Sequence
import tensorflow as tf
import numpy as np
import keras_transformer


def cumulative_segment_wrapper(fun):
    """Wrap a cumulative function such that it can be applied to segments.

    Args:
        fun: The cumulative function

    Returns:
        Wrapped function.

    """
    def wrapped_segment_op(x, segment_ids, **kwargs):
        with tf.compat.v1.name_scope(None, default_name=fun.__name__+'_segment_wrapper', values=[x]):
            segments, _ = tf.unique(segment_ids)
            n_segments = tf.shape(segments)[0]
            output_array = tf.TensorArray(x.dtype, size=n_segments, infer_shape=False)

            def loop_cond(i, out):
                return i < n_segments

            def execute_cumulative_op_on_segment(i, out):
                segment_indices = tf.where(tf.equal(segment_ids, segments[i]))
                seg_begin = tf.reduce_min(segment_indices)
                seg_end = tf.reduce_max(segment_indices)
                segment_data = x[seg_begin:seg_end+1]
                out = out.write(i, fun(segment_data, **kwargs))
                return i+1, out

            i_end, filled_array = tf.while_loop(
                loop_cond,
                execute_cumulative_op_on_segment,
                loop_vars=(tf.constant(0), output_array),
                parallel_iterations=10,
                swap_memory=True
            )
            output_tensor = filled_array.concat()
            output_tensor.set_shape(x.get_shape())
            return output_tensor

    return wrapped_segment_op


def cumulative_mean(tensor):
    """Cumulative mean of a rank 2 tensor.

    Args:
        tensor: Input tensor

    Returns:
        Tensor with same shape as input but containing cumulative mean.

    """
    assert len(tensor.shape) == 2
    n_elements = tf.cast(tf.shape(tensor)[0], tensor.dtype)
    start = tf.constant(1, dtype=tensor.dtype)
    n_elements_summed = tf.range(start, n_elements+1, dtype=tensor.dtype)
    return tf.cumsum(tensor, axis=0) / tf.expand_dims(n_elements_summed, -1)


cumulative_segment_mean = cumulative_segment_wrapper(cumulative_mean)
cumulative_segment_sum = cumulative_segment_wrapper(tf.math.cumsum)

class PaddedToSegments(tf.keras.layers.Layer):
    """Convert a padded tensor with mask to a stacked tensor with segments."""

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])

    def call(self, inputs, mask):
        valid_observations = tf.where(mask)
        collected_values = tf.gather_nd(inputs, valid_observations)
        return collected_values, valid_observations[:, 0]


class SegmentAggregation(tf.keras.layers.Layer):
    def __init__(self, aggregation_fn='sum', cumulative=False):
        super().__init__()
        self.cumulative = cumulative
        self.aggregation_fn = self._get_aggregation_fn(aggregation_fn)

    def _get_aggregation_fn(self, aggregation_fn):
        if not self.cumulative:
            if aggregation_fn == 'sum':
                return tf.math.segment_sum
            elif aggregation_fn == 'mean':
                return tf.math.segment_mean
            elif aggregation_fn == 'max':
                return tf.math.segment_max
            else:
                raise ValueError('Invalid aggregation function')
        else:
            if aggregation_fn == 'sum':
                return cumulative_segment_wrapper(tf.math.cumsum)
            elif aggregation_fn == 'mean':
                return cumulative_segment_wrapper(cumulative_mean)
            elif aggregation_fn == 'max':
                raise ValueError('max aggregation function not supported with cumulative aggregation.')
            else:
                raise ValueError('Invalid aggregation function')

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, data, segment_ids):
        assert segment_ids is not None
        return self.aggregation_fn(data, segment_ids)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_time=20000, n_dim=10, **kwargs):
        self.max_time = max_time
        self.n_dim = n_dim
        self._num_timescales = self.n_dim // 2
        super().__init__(**kwargs)

    def get_timescales(self):
        # This is a bit hacky, but works
        timescales = self.max_time ** np.linspace(0, 1, self._num_timescales)
        return timescales

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.timescales = self.add_weight(
            'timescales',
            (self._num_timescales, ),
            trainable=False,
            initializer=tf.keras.initializers.Constant(self.get_timescales())
        )

    def __call__(self, times):
        scaled_time = times / self.timescales[None, None, :]
        signal = tf.concat(
            [
                tf.sin(scaled_time),
                tf.cos(scaled_time)
            ],
            axis=-1)
        return signal

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_dim)


class TransformerModel(tf.keras.Model):
    def __init__(self, output_activation, output_dims, n_dims, n_heads,
                 n_layers, dropout, attn_dropout, aggregation_fn,
                 max_timescale):
        self._config = {name: val for name, val in locals().items() if name not in ['self', '__class__']}
        super().__init__()
        self.positional_encoding = PositionalEncoding(max_timescale, n_dim=n_dims)
        self.demo_embedding = tf.keras.layers.Dense(n_dims, activation=None, name="demo_embedding")
        self.element_embedding = tf.keras.layers.Dense(n_dims, activation=None, name="element_embedding")

        if isinstance(output_dims, Sequence):
            # We have an online prediction scenario
            assert output_dims[0] is None
            self.return_sequences = True
            output_dims = output_dims[1]
        else:
            self.return_sequences = False
        self.add = tf.keras.layers.Add(name="add_pos_enc")
        self.transformer_blocks = []
        for i in range(n_layers):
            transformer_block = keras_transformer.TransformerBlock(
                n_heads, dropout, attn_dropout, activation='relu',
                use_masking=self.return_sequences, vanilla_wiring=True)
            transformer_block._name = f"transformer_{i}"
            self.transformer_blocks.append(transformer_block)
            setattr(self, f'transformer_{i}', transformer_block)
        self.to_segments = PaddedToSegments()
        self.aggregation = SegmentAggregation(aggregation_fn, cumulative=self.return_sequences)
        self.out_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_dims, activation='relu', name="out_mlp/dense1"),
                tf.keras.layers.Dense(output_dims, output_activation, name="out_mlp/dense2")
            ],
            name='out_mlp'
        )

    def build(self, input_shapes):
        demo, times, values, measurements, lengths = input_shapes
        self.positional_encoding.build(times)
        self.demo_embedding.build(demo)
        embedding_input = (None, values[-1] + measurements[-1])
        self.element_embedding.build(embedding_input)
        embedding_shape = self.element_embedding.compute_output_shape(embedding_input)

        self.add.build([embedding_shape, embedding_shape])

        for block in self.transformer_blocks:
            block.build(tuple(embedding_shape))

        self.to_segments.build(embedding_shape)
        segments = self.to_segments.compute_output_shape(embedding_shape)
        aggregated_output = (self.aggregation.compute_output_shape(segments))
        self.out_mlp.build(aggregated_output)

        self.built = True

    def call(self, inputs):
        demo, times, values, measurements, lengths = inputs
        transformed_times = self.positional_encoding(times)
        value_modality_embedding = tf.concat(
            (
                values,
                tf.cast(measurements, tf.float32)
            ),
            axis=-1
        )

        # Somehow eager execution and graph mode behave differently.
        # In graph mode legths has an additional dimension
        if len(lengths.get_shape()) == 2:
            lengths = tf.squeeze(lengths, -1)
        mask = tf.sequence_mask(lengths+1, name='mask')

        demo_embedded = self.demo_embedding(demo)
        embedded = self.element_embedding(value_modality_embedding)
        combined = self.add([transformed_times, embedded])
        combined = tf.concat(
            [tf.expand_dims(demo_embedded, 1), combined], axis=1)
        transformer_out = combined
        for block in self.transformer_blocks:
            transformer_out = block(transformer_out, mask=mask)

        collected_values, segment_ids = self.to_segments(transformer_out, mask)

        aggregated_values = self.aggregation(collected_values, segment_ids)
        output = self.out_mlp(aggregated_values)

        if self.return_sequences:
            # If we should return sequences, then we need to transform the
            # output back into a tensor of the right shape
            valid_observations = tf.cast(tf.where(mask), tf.int32)
            output = tf.scatter_nd(
                valid_observations,
                output,
                tf.concat([tf.shape(mask), tf.shape(output)[-1:]], axis=0)
            )
            # Cut of the prediction only based on demographics
            output = output[:, 1:]
            # Indicate that the tensor contains invalid values
            output._keras_mask = mask[:, 1:]

        return output

    def get_config(self):
        return self._config

    def data_preprocessing_fn(self):
        def add_time_dim(inputs, label):
            demo, times, values, measurements, lengths = inputs
            times = tf.expand_dims(times, -1)
            return (demo, times, values, measurements, lengths), label
        return add_time_dim
