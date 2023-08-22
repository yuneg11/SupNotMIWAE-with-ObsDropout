from collections.abc import Sequence, Mapping
from typing import TYPE_CHECKING, Union, Any

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import keras_nlp
import keras_transformer


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

if TYPE_CHECKING:  # Support auto-completion in IDEs.
    from keras.api._v2 import keras
    from tensorflow_probability.python.distributions import Normal, Bernoulli
else:
    from tensorflow import keras
    Normal = tfp.distributions.Normal
    Bernoulli = tfp.distributions.Bernoulli
    MultivariateNormalFullCovariance = tfp.distributions.MultivariateNormalFullCovariance
    MultivariateNormalTriL = tfp.distributions.MultivariateNormalTriL

from ..modules import GRUD, GRUDInput, GRUDState, DecayInterpolate, InterpolateInput

from .. import functional as F


__all__ = [
    "SupNotMIWAEModel",
]


def rbf_kernel(T, length_scale):
    xs = tf.range(T, dtype=tf.float32)
    xs_in = tf.expand_dims(xs, 0)
    xs_out = tf.expand_dims(xs, 1)
    distance_matrix = tf.math.squared_difference(xs_in, xs_out)
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = tf.math.exp(-distance_matrix_scaled)
    return kernel_matrix


def diffusion_kernel(T, length_scale):
    assert length_scale < 0.5, "length_scale has to be smaller than 0.5 for the "\
                               "kernel matrix to be diagonally dominant"
    sigmas = tf.ones(shape=[T, T]) * length_scale
    sigmas_tridiag = tf.linalg.band_part(sigmas, 1, 1)
    kernel_matrix = sigmas_tridiag + tf.eye(T)*(1. - length_scale)
    return kernel_matrix


def matern_kernel(T, length_scale):
    xs = tf.range(T, dtype=tf.float32)
    xs_in = tf.expand_dims(xs, 0)
    xs_out = tf.expand_dims(xs, 1)
    distance_matrix = tf.math.abs(xs_in - xs_out)
    distance_matrix_scaled = distance_matrix / tf.cast(tf.math.sqrt(length_scale), dtype=tf.float32)
    kernel_matrix = tf.math.exp(-distance_matrix_scaled)
    return kernel_matrix


def cauchy_kernel(T, sigma, length_scale): # T = time_length
    xs = tf.range(T, dtype=tf.float32)
    xs_in = tf.expand_dims(xs, 0)
    xs_out = tf.expand_dims(xs, 1)

    distance_matrix = tf.math.squared_difference(xs_in, xs_out)
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = tf.math.divide(sigma, (distance_matrix_scaled + 1.))

    alpha = 0.001
    eye = tf.eye(num_rows= tf.shape(kernel_matrix)[-1])
    return kernel_matrix + alpha * eye




def cumulative_segment_wrapper(fun):
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


class StackedTransformer(keras.Model):
    def __init__(self, num_layer, n_hidden, num_heads):
        super().__init__()
        self.num_layer  = num_layer
        self.n_hidden = n_hidden
        self.num_heads= num_heads
        self.encoder = [
            keras_nlp.layers.TransformerEncoder(intermediate_dim=self.n_hidden, num_heads=self.num_heads)
            for _ in range(self.num_layer)
        ]

    def call(self, x, padding_mask, attention_mask=None):
        val = x
        for model in self.encoder:
            val = model(val, padding_mask=padding_mask, attention_mask=attention_mask)
        return val


from ..modules.gru_d import exp_relu
from keras import backend


from ..modules.interpolate import CONSTANT_INIT
from keras import backend, constraints, initializers, regularizers
from keras.utils import tf_utils

class DecayCell(keras.layers.Layer):
    def __init__(self, num_outputs, activation, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.activation = activation
        self.use_bias = use_bias

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.num_outputs,),
            name="decay_kernel",
            initializer=initializers.get("zeros"),
            regularizer=regularizers.get(None),
            constraint=constraints.get(None),
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.num_outputs,),
                name="decay_bias",
                initializer=initializers.get(CONSTANT_INIT),
                regularizer=regularizers.get(None),
                constraint=constraints.get('non_neg'),
            )

        self.built=True

    def call(self, inputs):
        gamma =  inputs * self.kernel
        if self.use_bias:
            gamma = backend.bias_add(gamma, self.bias)
        gamma = self.activation(gamma)
        return gamma


def compute_causal_mask(inputs):
    input_shape = tf.shape(inputs)
    batch_size, sequence_length = input_shape[0], input_shape[1]
    i = tf.range(sequence_length)[:, tf.newaxis] # [seq , 1]
    j = tf.range(sequence_length) # [seq]
    mask = tf.cast(i >= j, dtype="int32")
    mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
    mult = tf.concat(
        [
            tf.expand_dims(batch_size, -1),
            tf.constant([1, 1], dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.tile(mask, mult)


def merge_padding_and_attention_mask(
    inputs,
    padding_mask,
    attention_mask,
):
    """Merge padding mask with users' customized mask.
    Args:
        inputs: the input sequence.
        padding_mask: the 1D padding mask, of shape
            [batch_size, sequence_length].
        attention_mask: the 2D customized mask, of shape
            [batch_size, sequence1_length, sequence2_length].
    Return:
        A merged 2D mask or None. If only `padding_mask` is provided, the
        returned mask is padding_mask with one additional axis.
    """
    mask = padding_mask
    if hasattr(inputs, "_keras_mask"):
        if mask is None:
            # If no padding mask is explicitly provided, we look for padding
            # mask from the input data.
            mask = inputs._keras_mask
        else:
            logging.warning(
                "You are explicitly setting `padding_mask` while the `inputs` "
                "have built-in mask, so the built-in mask is ignored."
            )
    if mask is not None:
        # Add an axis for broadcasting, the attention mask should be 2D
        # (not including the batch axis).
        mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
    if attention_mask is not None:
        attention_mask = tf.cast(attention_mask, dtype=tf.int32)
        if mask is None:
            return attention_mask
        else:
            return tf.minimum(
                mask[:, tf.newaxis, :],
                attention_mask,
            )
    return mask


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_time=20000, n_dim=10, **kwargs):
        self.max_time = max_time
        self.n_dim = n_dim
        self._num_timescales = self.n_dim # self.n_dim // 2 # self.n_dim //2
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
        scaled_time = times / self.timescales[None, None, :] # bs, ts, feature_size/2
        cos_mask = tf.cast(tf.range(self.n_dim) % 2, dtype = tf.float32)
        sin_mask = 1 - cos_mask
        signal = (tf.sin(scaled_time) * sin_mask + tf.cos(scaled_time) * cos_mask)
        return signal

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_dim)


class SupNotMIWAEModel(keras.Model):
    def __init__(
        self,
        output_activation,
        output_dims,
        n_train_latents: int = 10,
        n_train_samples: int = 1,
        n_test_latents: int = 20,
        n_test_samples: int = 30,
        n_hidden: int = 128,
        n_units: int = 128,
        z_dim: int = 32,
        observe_dropout: float = 0.,  # Supports feature-wise dropout
        impute_type: Literal["decay", "fdecay"] = "fdecay",
        min_latent_sigma: float = 0.,
        min_sigma: float = 0.,
        classifier_num: int = 4,
        num_layer: int = 4,
        num_heads: int = 2,
        classifier_hidden: int = 128,
        gp_prior: bool = False,
        kernel: str = 'cauchy',
        length_scale: float = 7.,
        aggregation_method: str = 'max'
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

        if isinstance(observe_dropout, Sequence):
            observe_dropout = tf.constant([[observe_dropout]], dtype=tf.float32)
        else:
            observe_dropout = tf.constant(min(1., max(0., observe_dropout)))

        self.num_heads = num_heads
        self.output_activation = output_activation
        self.output_dims = output_dims
        self.n_train_latents = n_train_latents
        self.n_train_samples = n_train_samples
        self.n_test_latents = n_test_latents
        self.n_test_samples = n_test_samples
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.z_dim = z_dim
        self.observe_dropout = observe_dropout
        self.impute_type = impute_type
        self.min_latent_sigma = min_latent_sigma
        self.min_sigma = min_sigma
        self.classifier_num = classifier_num
        self.num_layer = num_layer
        self.classifier_hidden = classifier_hidden

        ## gp_prior parameter ##
        self.gp_prior = gp_prior
        self.kernel_scales = 1
        self.length_scale = length_scale
        self.kernel = kernel
        self.sigma = 1.005
        self.aggregation_method = aggregation_method

    def build(self, input_shape):
        # See `call` for the expected shape of the inputs.
        _, times_shape, values_shape, _, _ = input_shape

        self.x_dim = values_shape[-1]
        self.time_dim = times_shape + 1

        # === Encoder ===

        self.encoder = StackedTransformer(num_layer = self.num_layer ,n_hidden=self.n_hidden, num_heads = self.num_heads) # 2 or 4 256 or 32
        self.embedding_encoder =  keras.layers.Conv1D(self.n_hidden, kernel_size=8, padding="same", activation=tf.nn.tanh, name="embedding")
        self.pos_encoding = PositionalEncoding(max_time=100, n_dim = self.n_hidden)  # keras_nlp.layers.SinePositionEncoding()
        self.pos_encoding.build(self.time_dim)

        min_softplus = lambda x: self.min_latent_sigma + (1 - self.min_latent_sigma) * tf.nn.softplus(x)
        self.encoder_mu    = keras.layers.Dense(self.z_dim, activation=None,         name="encoder/mu")
        self.encoder_sigma = keras.layers.Dense(self.z_dim, activation=min_softplus, name="encoder/sigma")

        # === Decoder ===

        self.embedding_decoder =  keras.layers.Conv1D(self.n_hidden, kernel_size=3, padding="same", activation=tf.nn.tanh, name="embedding2")
        self.decoder = StackedTransformer(num_layer=self.num_layer ,n_hidden=self.n_hidden, num_heads=self.num_heads) # 2 or 4 256 or 32
        self.pos_encoding2 = PositionalEncoding(100, self.n_hidden)  # keras_nlp.layers.SinePositionEncoding()
        self.pos_encoding2.build(self.time_dim)

        min_softplus = lambda x: self.min_sigma + (1 - self.min_sigma) * tf.nn.softplus(x)
        self.decoder_mu    = keras.layers.Dense(self.x_dim, activation=None,         name="decoder/mu")
        self.decoder_sigma = keras.layers.Dense(self.x_dim, activation=min_softplus, name="decoder/sigma")

        # === Classifier ===

        self.embedding_classifier = keras.layers.Conv1D(self.classifier_hidden, kernel_size=3, padding="same", activation=tf.nn.tanh, name="conv1/class")
        self.pos_encoding3 = PositionalEncoding(100,self.classifier_hidden)  # keras_nlp.layers.SinePositionEncoding()
        self.pos_encoding3.build(self.time_dim)

        self.classifier_transformer = StackedTransformer(num_layer = self.classifier_num ,n_hidden=self.classifier_hidden, num_heads = self.num_heads) # 2 or 4 256 or 32
        self.to_segments = PaddedToSegments()

        if self.return_sequences:
            agg = 'mean'
        else:
            agg = self.aggregation_method

        self.aggregation = SegmentAggregation(aggregation_fn=agg,cumulative=self.return_sequences)

        self.classifier_dense = keras.layers.Dense(self.output_dims, activation=self.output_activation, name="classifier/dense")

        # === Misc ===

        if not self.return_sequences:
            self.initial_encoder_mlp = keras.Sequential([
                keras.layers.Dense(self.n_hidden, activation=tf.nn.tanh, name="initial_encoder2/dense1"),
                keras.layers.Dense(self.x_dim,  activation=tf.nn.tanh, name="initial_encoder2/dense2"),
            ], name="initial_encode2r")

        if self.impute_type == "decay":
            self.interpolator = DecayInterpolate(name="interpolator", decay_constraint="non_neg")
        elif self.impute_type == "fdecay":
            self.interpolator = DecayCell(self.x_dim, activation=exp_relu, name="interpolator")

        self.mnar_encoder = keras.Sequential([
            keras.layers.Dense(self.n_hidden, activation=tf.nn.tanh, name="mnar_encoder/dense1"),
            keras.layers.Dense(self.x_dim,    activation=tf.nn.tanh, name="mnar_encoder/dense2"),
        ], name="mnar_encoder")

    def get_config(self):
        return self._config

    def reconstruct_loss(self, x_tilde, x, masks):
        return tf.reduce_sum(abs(x_tilde - x) * masks) / (tf.reduce_sum(masks) + 1e-9)

    def call(self, inputs, output=None, training=False, return_loss=False, return_aux=False):
        # statics:      [n_batch, static_dim]
        # times:        [n_batch, n_times]
        # values:       [n_batch, n_times, x_dim]
        # measurements: [n_batch, n_times, x_dim]
        # lengths:      [n_batch, 1]
        statics, times, values, measurements, lengths = inputs

        if training:
            n_samples = self.n_train_samples
            n_latents = self.n_train_latents
        else:
            n_samples = self.n_test_samples
            n_latents = self.n_test_latents

        # Preprocess
        if len(tf.shape(lengths)) == 2:
            lengths = tf.squeeze(lengths, axis=-1)                                                                      # [n_batch]

        if not training:
            output = None
        elif output is not None and len(tf.shape(output)) == 1:
            output = tf.expand_dims(output, axis=1)                                                                     # [n_batch, 1]

        # NOTE: missing_mask includes padding_mask
        x_obs = values                                                                                                  # [n_batch, n_times, x_dim]
        missing_mask = measurements                                                                                     # [n_batch, n_times, x_dim]
        padding_mask = tf.sequence_mask(lengths, maxlen=tf.shape(times)[-1])                                            # [n_batch, n_times]

        if not self.gp_prior:
            q_z = self.encode(times, x_obs, missing_mask, padding_mask)                                                     # [n_batch, n_times, z_dim] x 2
            z_samples = q_z.sample(n_latents)                                                                               # [n_latents, n_batch, n_times, z_dim]

            # Prior: p(z)
            p_z = Normal(loc=0., scale=1.)                                                                              # []

            # Decode: h(zₖ; θ) -> p(xₖᵐ | zₖ)
            p_x_tilde = self.decode(times, z_samples, padding_mask)                                                         # [n_latents, n_batch, n_times, x_dim]

        else:
            q_z = self.encode_transpose(times, x_obs, missing_mask, padding_mask)
            z_samples = q_z.sample(n_latents)                                                                           # [n_latents, n_batch, n_times, z_dim] # transposed z_dim,ntimes

            time_length = tf.shape(x_obs)[1]
            p_z = self._get_prior(time_length)
            p_x_tilde = self.decode(times, tf.transpose(z_samples,perm=[0, 1, 3, 2]), padding_mask)                     # [n_latents, n_batch, n_times, x_dim]

        # === Impute ===
        if not self.gp_prior:
            # log p(xᵒ | zₖ)
            log_p_x_obs_given_z = tf.reduce_sum(tf.where(                                                                   # [n_latents, n_batch, n_times]
                missing_mask,                                                         # [           n_batch, n_times, x_dim]
                p_x_tilde.log_prob(x_obs),                                            # [n_latents, n_batch, n_times, x_dim]
                0.,
            ), axis=-1)

            # log p(zₖ)
            log_p_z = tf.reduce_sum(tf.where(                                                                               # [n_latents, n_batch, n_times]
                tf.expand_dims(padding_mask, axis=-1),                                # [           n_batch, n_times,     1]
                p_z.log_prob(z_samples),                                              # [n_latents, n_batch, n_times, z_dim]
                0.,
            ), axis=-1)

            # log q(zₖ | xᵒ)
            log_q_z_given_x_obs = tf.reduce_sum(tf.where(                                                                   # [n_latents, n_batch, n_times]
                tf.expand_dims(padding_mask, axis=-1),                                # [           n_batch, n_times,     1]
                q_z.log_prob(z_samples),                                              # [n_latents, n_batch, n_times, z_dim]
                0.,
            ), axis=-1)

        else:
            log_p_z = p_z.log_prob(z_samples)         # p_z shape: [n_latents,n_batch,z_dim] # this p_z dont have time axis

            log_p_x_obs_given_z = tf.reduce_sum(tf.where(                                                                   # [n_latents, n_batch, n_times]
                missing_mask,                                                         # [           n_batch, n_times, x_dim]
                p_x_tilde.log_prob(x_obs),                                            # [n_latents, n_batch, n_times, x_dim]
                0.,
            ), axis=-1)

            log_q_z_given_x_obs = tf.reduce_sum(tf.where(                                                                   # [n_latents, n_batch, n_times]
                tf.expand_dims(padding_mask, axis=1),                                # [           n_batch, n_times,     1]
                q_z.log_prob(z_samples),                                              # [n_latents, n_batch, n_times, z_dim]
                0.,
            ), axis=-2)

        if self.return_sequences and not self.gp_prior:
            log_p_x_obs_given_z = tf.cumsum(log_p_x_obs_given_z, axis=-1)                                               # [n_latents, n_batch, n_times]
            log_p_z             = tf.cumsum(log_p_z, axis=-1)                                                           # [n_latents, n_batch, n_times]
            log_q_z_given_x_obs = tf.cumsum(log_q_z_given_x_obs, axis=-1)
        elif self.return_sequences and self.gp_prior:
            log_p_z =  tf.reduce_sum(log_p_z,axis=-1, keepdims=True)                                                    # [n_latents, n_batch, z_dim]
            log_p_x_obs_given_z = tf.cumsum(log_p_x_obs_given_z, axis=-1)                                               # [n_latents, n_batch, n_times]
            log_q_z_given_x_obs = tf.cumsum(log_q_z_given_x_obs, axis=-1)
        else:
            log_p_x_obs_given_z = tf.reduce_sum(log_p_x_obs_given_z, axis=-1, keepdims=True)                            # [n_latents, n_batch, 1]
            log_p_z             = tf.reduce_sum(log_p_z, axis=-1, keepdims=True)                                        # [n_latents, n_batch, 1]
            log_q_z_given_x_obs = tf.reduce_sum(log_q_z_given_x_obs, axis=-1, keepdims=True)                            # [n_latents, n_batch, 1]

        # Generate: xₖⱼᵐ ~ p(xₖᵐ | zₖ)
        x_tilde = p_x_tilde.sample(n_samples)                                                                           # [n_samples, n_latents, n_batch, n_times, x_dim]

        # Dropout
        drop_mask, log_m = self.generate_observe_dropout_mask(tf.shape(x_tilde), missing_mask, training=training)       # [(n_samples), (n_latents), n_batch, n_times, x_dim], [(n_samples), (n_latents), n_batch]

        # Impute: xₖⱼ' = xᵒ and xₖⱼᵐ
        x_impute = self.impute(times, x_obs, x_tilde, drop_mask, padding_mask)                                          # [(n_samples), (n_latents), n_batch, n_times, x_dim]

        # log p(s | xᵒ, xₖⱼᵐ, mₖⱼ)
        logits_miss = self.mnar_encoder(x_impute)                                                                   # [(n_samples), (n_latents), n_batch, n_times, x_dim]
        p_s = Bernoulli(logits=logits_miss)                                                                         # [(n_samples), (n_latents), n_batch, n_times, x_dim]

        log_p_s_given_x = tf.reduce_sum(tf.where(                                                                   # [(n_samples), (n_latents), n_batch, n_times]
            tf.expand_dims(padding_mask, axis=-1),              # [                          n_batch, n_times,     1]
            p_s.log_prob(tf.cast(drop_mask, dtype=tf.float32)), # [(n_samples), (n_latents), n_batch, n_times, x_dim]
            0.,
        ), axis=-1)

        if self.return_sequences:
            log_p_s_given_x = tf.cumsum(log_p_s_given_x, axis=-1)                                                       # [(n_samples), (n_latents), n_batch, n_times]
        else:
            log_p_s_given_x = tf.reduce_sum(log_p_s_given_x, axis=-1, keepdims=True)                                    # [(n_samples), (n_latents), n_batch, 1]

        # Classify: f(xᵒ, xₖⱼᵐ; ϕ) -> p(y | xᵒ, xₖⱼᵐ, mₖⱼ)
        log_p_y = self.classify(statics, times, x_impute, drop_mask, padding_mask, lengths, output=output)                       # [(n_samples), (n_latents), n_batch, (n_times), y_dim]
        # rₖⱼ = p(s | xᵒ, xₖⱼᵐ, mₖⱼ) p(xᵒ | zₖ) p(zₖ) p(mₖⱼ) / q(zₖ | xᵒ)
        log_r = tf.expand_dims(log_p_x_obs_given_z + log_m + log_p_z - log_q_z_given_x_obs, axis=-1)                    # [(n_samples), (n_latents), n_batch, (n_times), 1]
        # wₖⱼ = rₖⱼ / ∑ₖⱼ rₖⱼ
        log_w = tf.nn.log_softmax(log_r, axis=1)                                                                        # [(n_samples), (n_latents), n_batch, (n_times), 1]

        # p(y | xᵒ) ≈ Eⱼ[ ∑ₖ wₖⱼ p(y | xᵒ, xₖⱼᵐ, mₖⱼ) ]
        y_logit = tf.reduce_mean(tf.reduce_logsumexp(log_w + log_p_y, axis=1), axis=0)                                  # [n_batch, (n_times), y_dim]
        y_prob = tf.exp(y_logit)
        if not self.return_sequences:
            y_prob = tf.squeeze(y_prob, axis=-2)                                                                        # [n_batch, y_dim]

        n_lat = tf.math.log(tf.cast(tf.shape(log_p_y)[1], tf.float32))

        log_p_s_given_x = tf.expand_dims(log_p_s_given_x,-1)

        # R(xᵒ, y) = Eⱼ[ log 1 / K ⋅ ∑ₖ p(y | xᵒ, xₖⱼᵐ, mₖⱼ) p(s | xᵒ, xₖⱼᵐ, mₖⱼ) p(xᵒ | zₖ) p(zₖ) / q(zₖ | xᵒ) ]
        loss = -tf.reduce_mean(tf.reduce_logsumexp(log_p_y + log_r + log_p_s_given_x, axis=1) - n_lat)

        # ESS = Eⱼ[ 1 / ∑ₖ wₖⱼ² ]
        ess = tf.reduce_mean(1. / tf.exp(tf.reduce_logsumexp(log_w * 2, axis=1)))                                       # []

        # Prediction error = Eⱼ[ 1 / K ⋅ ∑ₖ p(y | xᵒ, xₖⱼᵐ) / q(z | xᵒ)]
        pred_error = -tf.reduce_mean(log_p_y)

        # Reconstruction error = p(xᵒ | z)
        recon_error = -tf.reduce_mean(log_p_x_obs_given_z)

        # Regularization = p(zₖ) / q(z | xᵒ)
        regular = -tf.reduce_mean(log_p_z - log_q_z_given_x_obs)

        # others
        prefix = "Train" if training else "Valid"
        aux = {
            "log_w": log_w,
            "x_impute": x_impute,
            "x_mu": p_x_tilde.loc,
            "x_sigma": p_x_tilde.scale,
            "metrics": {
                f"{prefix}/ess": ess,
                f"{prefix}/pred_err": pred_error,
                f"{prefix}/recon_err": recon_error,
                f"{prefix}/regular": regular
            }
        }

        if return_loss and return_aux:
            return y_prob, loss, aux
        elif return_loss:
            return y_prob, loss
        elif return_aux:
            return y_prob, aux
        else:
            return y_prob

    def encode(self, times, values, missing_mask, padding_mask):
        times = tf.expand_dims(times,-1) # bs, ts, 1

        causal_mask = compute_causal_mask(values)
        causal_mask = causal_mask[0]
        transformed_times = self.pos_encoding(times) # [0,0.7,0.9,0.....] or times
        value = self.embedding_encoder(values)
        value += transformed_times
        r = self.encoder(value, padding_mask = padding_mask,attention_mask = causal_mask)

        z_mu = self.encoder_mu(r)                                                                                       # [n_batch, n_times, z_dim]
        z_sigma = self.encoder_sigma(r)                                                                                 # [n_batch, n_times, z_dim]
        q_z = Normal(loc=z_mu, scale=z_sigma) # Normal(loc=z_mu, scale = 1.)                                            # [n_batch, n_times, z_dim]
        return q_z                                                                                                      # [n_batch, n_times, z_dim]

    def decode(self, times, z, padding_mask):
        shape = tf.shape(z)  # (n_latents, n_batch, n_times, z_dim)

        z = tf.reshape(z, shape=[shape[0] * shape[1], shape[2], shape[3]])                                              # [n_latents x n_batch, n_times, z_dim]
        padding_mask = tf.tile(padding_mask, multiples=[shape[0], 1])                                                   # [n_latents x n_batch, n_times]

        times = tf.expand_dims(times,-1)
        transformed_times = tf.tile(self.pos_encoding2(times),multiples=[shape[0],1,1])

        z = self.embedding_decoder(z)
        z = transformed_times + z
        causal_mask = compute_causal_mask(z)
        causal_mask = causal_mask[0]

        h = self.decoder(z, padding_mask=padding_mask, attention_mask=causal_mask)
        h = tf.reshape(h, shape=[shape[0], shape[1], shape[2], -1])

        x_tilde_mu = self.decoder_mu(h)                                                                                 # [n_latents, n_batch, n_times, x_dim]
        x_tilde_sigma = self.decoder_sigma(h)                                                                           # [n_latents, n_batch, n_times, x_dim]
        p_x_tilde = Normal(loc=x_tilde_mu, scale=x_tilde_sigma)                                                         # [n_latents, n_batch, n_times, x_dim]
        return p_x_tilde                                                                                                # [n_latents, n_batch, n_times, x_dim] x 2

    def generate_observe_dropout_mask(self, shape, missing_mask, training=False):
        if tf.reduce_any(self.observe_dropout > 0.) and training:
            p_m = Bernoulli(probs=(1. - self.observe_dropout))                                                          # [] | [x_dim]
            drop_mask = p_m.sample(shape if self.observe_dropout.ndim == 0 else shape[:-1])                             # [(n_samples), (n_latents), n_batch, n_times, x_dim]

            log_m = tf.reduce_sum(tf.where(                                                                             # [(n_samples), (n_latents), n_batch, n_times]
                missing_mask,                                      # [                          n_batch, n_times, x_dim]
                p_m.log_prob(drop_mask),                           # [(n_samples), (n_latents), n_batch, n_times, x_dim]
                0.,
            ), axis=-1)

            drop_mask = tf.cast(drop_mask, dtype=bool) & missing_mask                                                   # [(n_samples), (n_latents), n_batch, n_times, x_dim]

        else:
            drop_mask = tf.broadcast_to(missing_mask, shape)                                                            # [(n_samples), (n_latents), n_batch, n_times, x_dim]

            log_m = tf.zeros(shape[:4])                                                                                 # [(n_samples), (n_latents), n_batch, n_times]

        if self.return_sequences:
            log_m = tf.cumsum(log_m, axis=-1)                                                                           # [(n_samples), (n_latents), n_batch, n_times]
        else:
            log_m = tf.reduce_sum(log_m, axis=-1, keepdims=True)                                                        # [(n_samples), (n_latents), n_batch, 1]

        return drop_mask, log_m                                                                                         # [(n_samples), (n_latents), n_batch, n_times, x_dim], [(n_samples), (n_latents), n_batch, (n_times)]

    def impute(self, times, x_obs, x_tilde, missing_mask, padding_mask):
        # Interpolate obs and combine with generated missing
        if self.impute_type == "decay":
            # xₖⱼ' = xᵒ if x is observed else xₖⱼᵐ ~ p(xᵐ | z)
            x_comb = tf.where(                                                                                          # [(n_samples), (n_latents), n_batch, n_times, x_dim]
                missing_mask,                                      # [(n_samples), (n_latents), n_batch, n_times, x_dim]
                x_obs,                                             # [                          n_batch, n_times, x_dim]
                x_tilde,                                           # [(n_samples), (n_latents), n_batch, n_times, x_dim]
            )

            shape = tf.shape(x_comb)  # ((n_samples), (n_latents), n_batch, n_times, x_dim)
            n_tiles = shape[0] * shape[1]

            times = tf.expand_dims(tf.tile(times, multiples=[n_tiles, 1]), axis=-1)                                     # [(n_samples) x (n_latents) x n_batch, n_times, 1]
            x_comb = tf.reshape(x_comb, shape=[n_tiles * shape[2], shape[3], shape[4]])                                 # [(n_samples) x (n_latents) x n_batch, n_times, x_dim]
            missing_mask = tf.reshape(missing_mask, shape=[n_tiles * shape[2], shape[3], shape[4]])                     # [(n_samples) x (n_latents) x n_batch, n_times, x_dim]
            padding_mask = tf.tile(padding_mask, multiples=[n_tiles, 1])                                                # [(n_samples) x (n_latents) x n_batch, n_times]

            x_impute = self.interpolator(                                                                               # [(n_samples) x (n_latents) x n_batch, n_times, x_dim]
                InterpolateInput(values=x_comb, mask=missing_mask, times=times),
                mask=padding_mask,
            )

            x_impute = tf.reshape(x_impute, shape=[shape[0], shape[1], shape[2], shape[3], shape[4]])                   # [(n_samples), (n_latents), n_batch, n_times, x_dim]

        elif self.impute_type == "fdecay":
            x_comb = tf.where(                                                                                          # [(n_samples), (n_latents), n_batch, n_times, x_dim]
                missing_mask,                                      # [(n_samples), (n_latents), n_batch, n_times, x_dim]
                x_obs,                                             # [                          n_batch, n_times, x_dim]
                x_tilde,                                           # [(n_samples), (n_latents), n_batch, n_times, x_dim]
            )

            shape = tf.shape(x_comb)  # ((n_samples), (n_latents), n_batch, n_times, x_dim)
            times = tf.tile(tf.expand_dims(times, axis=-1), multiples=[1,1,self.x_dim]) # [n_batch, n_times, x_dim]
            times = tf.tile(times[tf.newaxis,tf.newaxis,:,:,:], multiples=[shape[0], shape[1], 1, 1, 1])
            xcomb_and_times = tf.stack([x_comb, times])
            x_forward_and_t_forward = self.fill_last(xcomb_and_times, missing_mask)
            t_delta = times - x_forward_and_t_forward[1]
            gamma = self.interpolator(t_delta) # [n_batch, n_times, x_dim]
            x_impute = tf.where(
                missing_mask,
                x_comb,
                gamma * x_forward_and_t_forward[0] + (1 - gamma) * x_comb,
            )

        else:
            # xₖⱼ' = xᵒ if x is observed else xₖⱼᵐ ~ p(xᵐ | z)
            x_impute = tf.where(                                                                                        # [(n_samples), (n_latents), n_batch, n_times, x_dim]
                missing_mask,                                      # [                          n_batch, n_times, x_dim]
                x_obs,                                             # [                          n_batch, n_times, x_dim]
                x_tilde,                                           # [(n_samples), (n_latents), n_batch, n_times, x_dim]
            )

        return x_impute                                                                                                 # [(n_samples), (n_latents), n_batch, n_times, x_dim]

    def classify(self, statics, times, x_impute, impute_mask, padding_mask, lengths, output=None):
        shape = tf.shape(x_impute)  # ((n_samples), (n_latents), n_batch, n_times, x_dim)
        n_tiles = shape[0] * shape[1]
        x_impute = tf.reshape(x_impute, shape=[n_tiles * shape[2], shape[3], shape[4]])                                 # [(n_samples) x (n_latents) x n_batch, n_times, x_dim]
        padding_mask = tf.tile(padding_mask, multiples=[n_tiles, 1])                                                    # [(n_samples) x (n_latents) x n_batch, n_times]

        times = tf.expand_dims(tf.tile(times, multiples=[n_tiles, 1]), axis=-1)                                     # [(n_samples) x (n_latents) x n_batch, n_times, 1]

        if self.return_sequences:
            causal_mask = compute_causal_mask(x_impute)
            causal_mask = causal_mask[0]
            x_impute = self.embedding_classifier(x_impute) +  self.pos_encoding3(times)
            embedded = self.classifier_transformer(x_impute,padding_mask,causal_mask)
            v = embedded
        else:
            times = tf.concat([times[:,0,:][:,None,:],times],axis=1)
            initial_state = tf.tile(self.initial_encoder_mlp(statics), multiples=[n_tiles, 1])                                  # [(n_samples) x (n_latents) x n_batch, state_dim]
            padding_mask = tf.sequence_mask(lengths+1, maxlen=tf.shape(times)[1]) # maxlen=tf.shape(times)[1]+1                                           # [n_batch, n_times]
            padding_mask = tf.tile(padding_mask, multiples=[n_tiles, 1])                                                    # [(n_samples) x (n_latents) x n_batch, n_times]
            x_impute = tf.concat((initial_state[:,None,:],x_impute),axis=1)

            x_impute = self.embedding_classifier(x_impute) + self.pos_encoding3(times) # physionet 고성능은 pos enc 없었음
            embedded = self.classifier_transformer(x_impute,padding_mask)

            collected_values, segment_ids = self.to_segments(embedded, padding_mask)
            v = self.aggregation(collected_values, segment_ids)

        probs = self.classifier_dense(v)                                                                                # [(n_samples) x (n_latents) x n_batch, (n_times,) y_dim]
        p_y = Bernoulli(probs=probs)                                                                                    # [(n_samples) x (n_latents) x n_batch, (n_times,) y_dim]
        if self.return_sequences:
            if output is None:
                labels = tf.ones_like(probs)                                                                            # [(n_samples) x (n_latents) x n_batch, n_times, y_dim]
            else:
                labels = tf.cast(tf.tile(output, multiples=[n_tiles, 1, 1]), dtype=tf.float32)                          # [(n_samples) x (n_latents) x n_batch, n_times, y_dim]

            log_p_y = p_y.log_prob(labels)                                                                              # [(n_samples) x (n_latents) x n_batch, n_times, y_dim]
            log_p_y = tf.reshape(log_p_y, shape=(shape[0], shape[1], shape[2], shape[3], self.output_dims))             # [(n_samples),  (n_latents),  n_batch, n_times, y_dim]

        else:
            if output is None:
                labels = tf.ones_like(probs)                                                                            # [(n_samples) x (n_latents) x n_batch, y_dim]
            else:
                labels = tf.cast(tf.tile(output, multiples=[n_tiles, 1]), dtype=tf.float32)                             # [(n_samples) x (n_latents) x n_batch, y_dim]

            log_p_y = p_y.log_prob(labels)                                                                              # [(n_samples) x (n_latents) x n_batch, y_dim]
            log_p_y = tf.reshape(log_p_y, shape=(shape[0], shape[1], shape[2], 1, self.output_dims))                    # [(n_samples),  (n_latents),  n_batch, 1, y_dim]

        return log_p_y                                                                                                  # [(n_samples), (n_latents), n_batch, (n_times), y_dim]

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            _, loss, aux = self(x, output=y, training=True, return_loss=True, return_aux=True)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss, **aux.get("metrics", {})}

    def test_step(self, data):
        x, y = data
        _, loss, aux = self(x, training=False, return_loss=True, return_aux=True)
        return {"loss": loss, **aux.get("metrics", {})}

    def fill_last(self, a, mask, seq_axis=-2):
        a_ndim = len(a.shape)
        mask_ndim = len(mask.shape)
        if seq_axis >= 0 and a_ndim != mask_ndim:
            raise ValueError("a and mask must have the same ndim if seq_axis is >= 0")

        a_seq_axis = (a_ndim + seq_axis) if seq_axis < 0 else seq_axis
        mask_seq_axis = (mask_ndim + seq_axis) if seq_axis < 0 else seq_axis

        a_tr_idxs = [a_seq_axis] + [i for i in range(a_ndim) if i != a_seq_axis]
        mask_tr_idxs = [mask_seq_axis] + [i for i in range(mask_ndim) if i != mask_seq_axis]

        v_shape = tf.broadcast_dynamic_shape(tf.shape(a), tf.shape(mask))
        v_seq_axis = (tf.size(v_shape) + seq_axis) if seq_axis < 0 else seq_axis
        # v_tr_idxs = list(range(1, v_seq_axis + 1)) + [0] + list(range(v_seq_axis + 1, tf.size(v_shape)))
        v_tr_idxs = (1,2,3,4,0,5)

        scan_shape = tf.concat((v_shape[:v_seq_axis], v_shape[v_seq_axis + 1:]), axis=0)

        v = tf.transpose(tf.scan(
            lambda last, m_and_v: tf.where(m_and_v[0], m_and_v[1], last),
            elems=(tf.transpose(mask, mask_tr_idxs), tf.transpose(a, a_tr_idxs)),
            initializer=tf.zeros(scan_shape, dtype=a.dtype),
        ), v_tr_idxs)

        return v


    def data_preprocessing_fn(self):
        def one_hot_label(ts, labels):      # batch 축 존재 x
            # Ignore demographics for now
            demo, X, Y, measurements, lengths = ts
            # labels = [50] -> [50,11]
            if self.return_sequences:
                depth = 11
                labels = tf.one_hot(labels,depth)
            else:
                max_time = tf.math.reduce_max(X)
                X = tf.math.divide_no_nan(X, max_time)
                pass

            return (demo, X, Y, measurements, lengths), labels

        return one_hot_label

    def _get_prior(self,time_length):
        # Compute kernel matrices for each latent dimension
        kernel_matrices = []
        for i in range(self.kernel_scales):
            if self.kernel == "rbf":
                kernel_matrices.append(rbf_kernel(time_length, self.length_scale / 2**i))
            elif self.kernel == "diffusion":
                kernel_matrices.append(diffusion_kernel(time_length, self.length_scale / 2**i))
            elif self.kernel == "matern":
                kernel_matrices.append(matern_kernel(time_length, self.length_scale / 2**i))
            elif self.kernel == "cauchy":
                kernel_matrices.append(cauchy_kernel(time_length, self.sigma, self.length_scale / 2**i))

        # Combine kernel matrices for each latent dimension
        tiled_matrices = []
        total = 0
        for i in range(self.kernel_scales): # usually kernel scales = 1
            if i == self.kernel_scales-1:
                multiplier = self.z_dim - total
            else:
                multiplier = int(tf.math.ceil(self.z_dim / self.kernel_scales))
                total += multiplier
            tiled_matrices.append(tf.tile(tf.expand_dims(kernel_matrices[i], 0), [multiplier, 1, 1]))
        kernel_matrix_tiled = tf.concat(tiled_matrices,0) # np.concatenate(tiled_matrices)
        assert len(kernel_matrix_tiled) == self.z_dim
        # tf.print(tf.shape(kernel_matrix_tiled))
        # print(kernel_matrix_tiled[0]==kernel_matrix_tiled[1]) true

        prior = MultivariateNormalFullCovariance(
            loc=tf.zeros([self.z_dim, time_length], dtype=tf.float32),
            covariance_matrix=kernel_matrix_tiled) # [z_dim,length,length]
        return prior

    def encode_transpose(self, times, values, missing_mask, padding_mask):
        times = tf.expand_dims(times,-1) # bs, ts, 1
        #value_modality_embedding =  self.embedding(values) # self.embedding(tf.concat((values,tf.cast(missing_mask, tf.float32),times),axis=-1))
        #value = value_modality_embedding

        causal_mask = compute_causal_mask(values)
        causal_mask = causal_mask[0]
        transformed_times = self.pos_encoding(times) # [0,0.7,0.9,0.....] or times
        value = self.embedding_encoder(values)
        value += transformed_times
        r = self.encoder(value, padding_mask = padding_mask,attention_mask = causal_mask)
        # r이랑 classifier에서 나온 embedded랑 l2 norm 줄이기

        z_mu = self.encoder_mu(r)                                                                                       # [n_batch, n_times, z_dim]
        z_sigma = self.encoder_sigma(r)                                                                                 # [n_batch, n_times, z_dim]
        z_mu = tf.transpose(z_mu,perm=[0,2,1])
        z_sigma = tf.transpose(z_sigma,perm=[0,2,1])
        q_z = Normal(loc=z_mu, scale=z_sigma)
        return q_z                                                                                                      # [n_batch, n_times, z_dim]
