from collections.abc import Sequence, Mapping
from random import betavariate

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import TYPE_CHECKING, Union, Any

# from torch import alpha_dropout

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

if TYPE_CHECKING:  # Support auto-completion in IDEs.
    from keras.api._v2 import keras
    from tensorflow_probability.python.distributions import Normal, Bernoulli, MultivariateNormalFullCovariance,MultivariateNormalTriL
else:
    from tensorflow import keras
    Normal = tfp.distributions.Normal
    Bernoulli = tfp.distributions.Bernoulli
    MultivariateNormalFullCovariance = tfp.distributions.MultivariateNormalFullCovariance
    MultivariateNormalTriL = tfp.distributions.MultivariateNormalTriL

from ..modules import GRUD, GRUDInput, GRUDState, DecayInterpolate, InterpolateInput

from .. import functional as F


__all__ = [
    "GPVAEModel",
]

''' 
GP kernel functions 
'''


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
    # print(tf.shape(kernel_matrix)[-1])
    eye = tf.eye(num_rows= tf.shape(kernel_matrix)[-1]) #tf.eye(num_rows=kernel_matrix.shape.as_list()[-1])
    return kernel_matrix + alpha * eye

class GPVAEModel(keras.Model):
    def __init__(
        self,
        n_hidden: int = 64,
        z_dim: int = 32,
        dropout: float = 0.,
        encoder_type: Literal["mlp", "cnn", "gru", "grud"] = "cnn",
        decoder_type: Literal["mlp", "cnn", "gru", "grud"] = "mlp",
        min_latent_sigma: float = 0.,
        min_sigma: float = 0.,
        objective: str = "elbo",
        alpha: float = 1.,
        beta: float = 1.,
        gamma: float = 1.,
        kernel="cauchy",
        length_scale: float = 1.,
        kernel_scales: int = 1,
        n_latents: int = 1,
        encoder_dist: Literal["joint","diag"] = "diag"
    ):
        self._config = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        super().__init__()

        self.kernel = kernel
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales
        self.n_hidden = n_hidden
        self.z_dim = z_dim
        self.dropout = min(1., max(0., dropout))
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.min_latent_sigma = min_latent_sigma
        self.min_sigma = min_sigma
        self.alpha= alpha
        self.beta= beta
        self.gamma = gamma
        self.n_latents = n_latents
        self.sigma = 1.005
        self.encoder_dist = encoder_dist
    def build(self, input_shape):
        # See `call` for the expected shape of the inputs.
        _, _, values_shape, _, _ = input_shape

        self.x_dim = values_shape[-1]

        # === Encoder ===

        if self.encoder_type == "mlp":
            self.encoder = keras.Sequential([
                keras.layers.Dense(self.n_hidden, activation=tf.nn.tanh, name="encoder/dense1"),
                keras.layers.Dense(self.n_hidden, activation=tf.nn.tanh, name="encoder/dense2"),
            ], name="encoder")
        elif self.encoder_type == "cnn":
            self.encoder = keras.Sequential([
                keras.layers.Conv1D(self.n_hidden, kernel_size=32, padding="same", activation=tf.nn.tanh, name="encoder/conv1"),
                keras.layers.Dense(self.n_hidden, activation=tf.nn.tanh, name="encoder/dense2")
            ], name="encoder")

        min_softplus = lambda x: self.min_latent_sigma + (1 - self.min_latent_sigma) * tf.nn.softplus(x)
        
        self.encoder_mu    = keras.layers.Dense(self.z_dim, activation=None,         name="encoder/mu")
        if self.encoder_dist=='joint':
            self.encoder_sigma = keras.layers.Dense(2*self.z_dim, activation=tf.nn.sigmoid, name="encoder/sigma") #for  banded encoder  - we can use softplus
        else:
            self.encoder_sigma = keras.layers.Dense(self.z_dim, activation=tf.nn.sigmoid, name="encoder/sigma") #for  banded encoder  - we can use softplus

        # === Decoder ===

        if self.decoder_type == "mlp":
            self.decoder = keras.Sequential([
                keras.layers.Dense(256, activation=tf.nn.tanh, name="decoder/dense1"),
                keras.layers.Dense(256, activation=tf.nn.tanh, name="decoder/dense2"),
            ], name="decoder")
            self.decoder_sigma = keras.layers.Dense(self.x_dim, activation=tf.nn.sigmoid, name="decoder/sigma") #for  banded encoder  - we can use softplus

        min_softplus = lambda x: self.min_sigma + (1 - self.min_sigma) * tf.nn.softplus(x)
        self.decoder_mu    = keras.layers.Dense(self.x_dim, activation=None,         name="decoder/mu")
        # self.decoder_sigma = keras.layers.Dense(self.x_dim, activation=min_softplus, name="decoder/sigma") # for gpvae, they set var =1
        self.decoder_sigma = keras.layers.Dense(self.x_dim, activation=tf.nn.sigmoid, name="decoder/sigma") #for  banded encoder  - we can use softplus

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

        # Preprocess
        if len(tf.shape(lengths)) == 2:
            lengths = tf.squeeze(lengths, axis=-1)                                                                      # [n_batch]

        # NOTE: missing_mask includes padding_mask
        x_obs = values                                                                                                  # [n_batch, n_times, x_dim]
        missing_mask = measurements                                                                                     # [n_batch, n_times, x_dim]
        padding_mask = tf.sequence_mask(lengths)                                                                        # [n_batch, n_times]
        time_length = tf.shape(x_obs)[1]
        batch_size = tf.shape(x_obs)[0]
        if self.encoder_dist=='joint':
            q_z = self.encode(times, x_obs, missing_mask, padding_mask,time_length,batch_size)                                                     # [n_batch, n_times, z_dim] x 2 # [nbatch,z_dim,ntime] (transpose)
        else:
            q_z = self.encode2(times, x_obs, missing_mask, padding_mask,time_length,batch_size)                                                     # [n_batch, n_times, z_dim] x 2 # [nbatch,z_dim,ntime] (transpose)

        # Latent: zₖ ~ q(z | xᵒ)
        
        z_samples = q_z.sample(self.n_latents)                                                                               # [n_latents, n_batch, n_times, z_dim] # transposed z_dim,ntimes
        #tf.print(tf.shape(z_samples))
        # time_length = tf.shape(x_obs)[1]
        # print(z_samples.shape)
        p_z = self._get_prior(time_length)
        # Decode: h(zₖ; θ) -> p(xₖᵐ | zₖ)
        # a=p_z.sample((10,10)) # [10,10,z_dim,time_length]
        p_x_tilde = self.decode(times, tf.transpose(z_samples,perm=[0,1,3,2]), padding_mask)                                                         # [n_latents, n_batch, n_times, x_dim]
        n_samples=50 # 1 or 10
        x_tilde = p_x_tilde.sample(n_samples)
        # === Impute ===

        #tf.print(tf.shape(tf.expand_dims(padding_mask, axis=1)))
        #tf.print(tf.shape(p_z.log_prob(z_samples)))
        
        log_p_z = p_z.log_prob(z_samples) # [1,128,35]
        
        
        # log p(xᵒ | zₖ)
        log_p_x_obs_given_z = tf.reduce_sum(tf.where(                                                                   # [n_latents, n_batch, n_times]
            missing_mask,                                                         # [           n_batch, n_times, x_dim]
            p_x_tilde.log_prob(x_obs),                                            # [n_latents, n_batch, n_times, x_dim]
            0.,
        ), axis=-1)

        # # log p(zₖ)
        # log_p_z = tf.reduce_sum(tf.where(                                                                               # [n_latents, n_batch, n_times]
        #     tf.expand_dims(padding_mask, axis=1),                                # [           n_batch, 1, n_times]
        #     p_z.log_prob(z_samples),                                              # [n_latents, n_batch, z_dim,n_times]
        #     0.,
        # ), axis=-2)
        # print(q_z.log_prob(z_samples).shape) # [1,128,35]
        # log q(zₖ | xᵒ)
        if self.encoder_dist=='joint':
            log_q_z_given_x_obs = q_z.log_prob(z_samples ) # 
        else:
            log_q_z_given_x_obs = tf.reduce_sum(tf.where(                                                                   # [n_latents, n_batch, n_times]
                tf.expand_dims(padding_mask, axis=1),                                # [           n_batch, n_times,     1]
                q_z.log_prob(z_samples),                                              # [n_latents, n_batch, n_times, z_dim]
                0.,
            ), axis=-2)

        log_p_x_obs_given_z = tf.reduce_sum(log_p_x_obs_given_z, axis=-1, keepdims=True)                            # [n_latents, n_batch, 1]
        log_p_z             = tf.reduce_sum(log_p_z, axis=-1, keepdims=True)                                        # [n_latents, n_batch, 1]
        log_q_z_given_x_obs = tf.reduce_sum(log_q_z_given_x_obs, axis=-1, keepdims=True)                            # [n_latents, n_batch, 1]

        # wₗ = softmax( p(xᵒ | zₖ) p(z) / q(z | xᵒ) )
        # log_w_latents = tf.nn.log_softmax(log_p_x_obs_given_z + log_p_z - log_q_z_given_x_obs, axis=0)                  # [n_latents, n_batch, (n_times)]

        # Generate: xₖⱼᵐ ~ p(xₖᵐ | zₖ)
        x_tilde = p_x_tilde.sample(n_samples)         # [(n_samples), (n_latents), n_batch, n_times, x_dim]

        # Dropout
        # Impute: xₖⱼ' = xᵒ and xₖⱼᵐ
        x_impute =  tf.where(                                                                                        # [(n_samples), (n_latents), n_batch, n_times, x_dim]
            missing_mask,                                      # [                          n_batch, n_times, x_dim]
            x_obs,                                             # [                          n_batch, n_times, x_dim]
            x_tilde)                                           # [(n_samples), (n_latents), n_batch, n_times, x_dim])                                          # [(n_samples), (n_latents), n_batch, n_times, x_dim]

        log_r = tf.expand_dims(log_p_x_obs_given_z + self.beta*(log_p_z - log_q_z_given_x_obs), axis=-1) 

        loss = -tf.reduce_mean(log_r)

        y_prob = tf.ones(shape=(tf.shape(values)[0],1))

        # ESS = Eⱼ[ 1 / ∑ₖ wₖⱼ² ]

        # Prediction error = Eⱼ[ 1 / K ⋅ ∑ₖ p(y | xᵒ, xₖⱼᵐ) / q(z | xᵒ)]
        pred_error = 0.# -tf.reduce_mean(tf.reduce_logsumexp(log_p_y + tf.expand_dims(log_q_z_given_x_obs, axis=-1), axis=1) - n_lat)

        # Reconstruction error = p(xᵒ | z)
        recon_error = -tf.reduce_mean(log_p_x_obs_given_z)

        # Regularization = p(zₖ) / q(z | xᵒ)
        regular = -tf.reduce_mean(log_p_z - log_q_z_given_x_obs)
        # others
        prefix = "Train" if training else "Valid"
        aux = {
            "x_impute": x_impute,
            "x_mu": p_x_tilde.loc,
            "x_sigma": p_x_tilde.scale,
            "metrics": {
                f"{prefix}/pred_err": pred_error,
                f"{prefix}/recon_err": recon_error,
                f"{prefix}/regular": regular,
                f"{prefix}/loss": loss
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
    
    def encode(self, times, values, missing_mask, padding_mask,time_length,batch_size,transpose=True): # banded joint encoder
        
        
        r = self.encoder(values)                                                                                    # [n_batch, n_times, n_hidden]
        z_mu = self.encoder_mu(r)                                                                                       # [n_batch, n_times, z_dim]
        z_sigma = self.encoder_sigma(r)                                                                                 # [n_batch, n_times, z_dim]
        if transpose:
            z_mu = tf.transpose(z_mu,perm=[0,2,1])
            z_sigma = tf.transpose(z_sigma,perm=[0,2,1])

            mapped_reshaped = tf.reshape(z_sigma, [batch_size, self.z_dim, 2*time_length])

            dense_shape = [batch_size, self.z_dim, time_length, time_length]
            idxs_1 = np.repeat(np.arange(batch_size), self.z_dim*(2*time_length-1))
            idxs_2 = np.tile(np.repeat(np.arange(self.z_dim), (2*time_length-1)), batch_size)
            idxs_3 = np.tile(np.concatenate([np.arange(time_length), np.arange(time_length-1)]), batch_size*self.z_dim)
            idxs_4 = np.tile(np.concatenate([np.arange(time_length), np.arange(1,time_length)]), batch_size*self.z_dim)
            idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)
            # with tf.device('/cpu:0'):
                # Obtain covariance matrix from precision one
            mapped_values = tf.reshape(mapped_reshaped[:, :, :-1], [-1])
            prec_sparse = tf.sparse.SparseTensor(indices=idxs_all, values=mapped_values, dense_shape=dense_shape)
            prec_sparse = tf.sparse.reorder(prec_sparse)
            prec_tril = tf.compat.v1.sparse_add(tf.zeros(prec_sparse.dense_shape, dtype=tf.float32), prec_sparse)
            eye = tf.eye(num_rows=prec_tril.shape.as_list()[-1], batch_shape=prec_tril.shape.as_list()[:-2])
            prec_tril = prec_tril + eye
            cov_tril = tf.linalg.triangular_solve(matrix=prec_tril, rhs=eye, lower=False)
            cov_tril = tf.where(tf.math.is_finite(cov_tril), cov_tril, tf.zeros_like(cov_tril))

            num_dim = len(cov_tril.shape)
            perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
            cov_tril_lower = tf.transpose(cov_tril, perm=perm)
            z_dist = MultivariateNormalTriL(loc=z_mu, scale_tril=cov_tril_lower)

            return z_dist
        else:
            q_z = Normal(loc=z_mu, scale=z_sigma)                                                                           # [n_batch, n_times, z_dim]
            return q_z                                                                                                      # [n_batch, n_times, z_dim]

    def decode(self, times, z, padding_mask):
        if self.decoder_type == "mlp":
            h = self.decoder(z)                                                                                         # [n_latents, n_batch, n_times, n_hidden]
        elif self.decoder_type == "cnn":
            h = self.decoder(z)                                                                                         # [n_latents, n_batch, n_times, n_hidden]

        x_tilde_mu = self.decoder_mu(h)                                                                                 # [n_latents, n_batch, n_times, x_dim]
        x_tilde_sigma = self.decoder_sigma(h)                                                                           # [n_latents, n_batch, n_times, x_dim]
        # x_tilde sigma will be 1
        p_x_tilde = Normal(loc=x_tilde_mu, scale= x_tilde_sigma)          # scale = 1                                               # [n_latents, n_batch, n_times, x_dim]
        return p_x_tilde                                                                                                # [n_latents, n_batch, n_times, x_dim] x 2


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
                multiplier = int(np.ceil(self.z_dim / self.kernel_scales))
                total += multiplier
            tiled_matrices.append(tf.tile(tf.expand_dims(kernel_matrices[i], 0), [multiplier, 1, 1]))
        kernel_matrix_tiled = tf.concat(tiled_matrices,0) # np.concatenate(tiled_matrices)
        assert len(kernel_matrix_tiled) == self.z_dim
        # tf.print(kernel_matrix_tiled.shape)
        prior = MultivariateNormalFullCovariance(
            loc=tf.zeros([self.z_dim, time_length], dtype=tf.float32),
            covariance_matrix=kernel_matrix_tiled) # [z_dim,length,length]
        return prior

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


    def encode2(self, times, values, missing_mask, padding_mask,time_length,batch_size,transpose=True): # banded joint encoder
                    
        r = self.encoder(values)                                                                                    # [n_batch, n_times, n_hidden]
        z_mu = self.encoder_mu(r)                                                                                       # [n_batch, n_times, z_dim]
        z_sigma = self.encoder_sigma(r)                                                                                 # [n_batch, n_times, z_dim]
        if transpose:
            z_mu = tf.transpose(z_mu,perm=[0,2,1])
            z_sigma = tf.transpose(z_sigma,perm=[0,2,1])
            q_z = Normal(loc=z_mu, scale=z_sigma)  
            return q_z
        else:
            q_z = Normal(loc=z_mu, scale=z_sigma)  
            return q_z