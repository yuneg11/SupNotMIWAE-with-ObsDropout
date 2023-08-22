from tensorflow.python.keras.engine.base_layer import Layer
import numpy as np
from collections.abc import Sequence, Mapping
from einops import rearrange
import tensorflow as tf
import tensorflow_probability as tfp
from typing import TYPE_CHECKING, Union, Any
import keras_nlp
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

if TYPE_CHECKING:  # Workaround for VS Code intellisense
    # from tensorflow.python import keras
    from keras.api._v2 import keras
    from tensorflow_probability.python.distributions import Normal, Bernoulli
else:
    keras = tf.keras
    Normal = tfp.distributions.Normal
    Bernoulli = tfp.distributions.Bernoulli

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

        ## debugged
        cos_mask = tf.cast(tf.range(self.n_dim) % 2, dtype = tf.float32)
        sin_mask = 1 - cos_mask
        signal = (tf.sin(scaled_time) * sin_mask + tf.cos(scaled_time) * cos_mask)

        # original
        # signal = tf.concat(
        #     [
        #         tf.sin(scaled_time),
        #         tf.cos(scaled_time)
        #     ],
        #     axis=-1)    # 이렇게 하면 짝수 피쳐에 sin,cos가 번갈아가면서 들어가는게 아니고 sin cos가 그냥 반반씩 받아서 concat되는거 아님?
        return signal

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_dim)




class stacked_transformer(keras.Model):

    def __init__(self,num_layer,n_hidden,num_heads):
        super().__init__()
        self.num_layer  = num_layer
        self.n_hidden = n_hidden
        self.num_heads= num_heads
        self.encoder = [keras_nlp.layers.TransformerEncoder(intermediate_dim=self.n_hidden, num_heads=self.num_heads) for _ in range(self.num_layer)] # num_heads =4 
    def call(self,x,padding_mask,attention_mask=None):
        val = x
        for model in self.encoder:
            val = model(val,padding_mask=padding_mask, attention_mask = attention_mask)

        #value = keras.layers.Dense(self.n_hidden)(val)
        return val



# class EncoderLayer(keras.layers.Layer):
#     def __init__(self, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, attn_dropout=0.1):
#         super(EncoderLayer, self).__init__()

#         self.diagonal_attention_mask = True
#         self.d_feature = d_feature

#         self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6) # sgiykd cgabge
#         self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
#         self.dropout = keras.layers.Dropout(rate=dropout)
#         self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

#     def call(self, enc_input,mask): # 이부분에 length 추가하기

#         residual = enc_input
#         # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
#         enc_input = self.layer_norm(enc_input)
#         enc_output, attn_weights = self.slf_attn(enc_input, enc_input, enc_input,mask) # 여기에 lengths넣기
#         enc_output = self.dropout(enc_output)
#         enc_output += residual

#         enc_output = self.pos_ffn(enc_output)
#         return enc_output, attn_weights


class SAITSMODEL(keras.Model):

    def __init__(
        self,
        output_activation,
        output_dims,
        n_hidden: int = 128,
        d_model:int = 256,
        num_layer: int = 4,
        n_head:int = 4,
        dropout: float = 0.0,
        train_type: str = 'imputation',
        MIT: bool = False,
        input_with_mask: bool = True,
        MIT_missing_rate: float = 0.2
        ): # params추가해야됨
        super().__init__()
        self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }
        self.output_activation = output_activation
        self.output_dims = output_dims
        self.num_layer   = num_layer
        self.d_model = d_model
        self.n_head = n_head
        self.dropout_rate=dropout
        self.input_with_mask = input_with_mask
        self.MIT = MIT
        self.MIT_missing_rate = MIT_missing_rate
        self.add = keras.layers.Add()
        self.train_type = train_type
        self.n_hidden = n_hidden
        self.pos_encoding = PositionalEncoding(max_time = 100 , n_dim = self.d_model)  # keras_nlp.layers.SinePositionEncoding()


    def preprocess(self,measurements,artificial_mis): # artificial masking 여기서 포함

        artificial_mask = tfp.distributions.Bernoulli(probs=1-artificial_mis).sample(sample_shape=tf.shape(measurements))
        masks = tf.cast(measurements, tf.float32)*tf.cast(artificial_mask,tf.float32) # hat(M)에 해당 - artificially masked된 자리까지 missing으로 보는 mask
        indicator_masks = (1.-masks)*tf.cast(measurements, tf.float32) ## make indicator - artificially masked된 자리만 1로 찍힌 mask
        return masks,indicator_masks


    def data_preprocessing_fn(self):  # 전부 length 300으로 패딩하기, batch 차원이 없음  여기는
        def add_time_dim(inputs, label):
            demo, times, values, measurements, lengths = inputs # (none,16)
            times = tf.expand_dims(times, -1)
            return (demo, times, values, measurements, lengths), label
        return add_time_dim

    def build(self, input_shape):
        _, times_shape, values_shape, _, _ = input_shape
        self.d_feature = values_shape[-1]
        self.actual_d_feature = self.d_feature * 2 if self.input_with_mask else self.d_feature
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)
        self.transformer1 =  stacked_transformer(num_layer = self.num_layer, n_hidden = self.n_hidden, num_heads=self.n_head )
        self.transformer2 =  stacked_transformer(num_layer = self.num_layer, n_hidden = self.n_hidden, num_heads=self.n_head )

        # self.position_enc = PositionalEncoding(d_model, n_position=200) <- 제대로 구현해서 살려서 쓰기
        # for operation on time dim
        self.embedding_1 = keras.layers.Dense(self.d_model,name='embed1')
        self.pos_encoding.build(times_shape)
        self.reduce_dim_z =  keras.layers.Dense(self.d_feature,name='reduce_z_1')

        self.embedding_2 =  keras.layers.Dense(self.d_model,name='embed2')
        self.reduce_dim_beta = keras.layers.Dense(self.d_feature,name='reduce_beta')
        self.reduce_dim_gamma = keras.layers.Dense(self.d_feature,name='reduce_gamma')
            
    def call(self,inputs,training=False, output=None,return_aux=False,return_loss=False):  # equivalent to impute and calc loss
        statics, times, values, measurements, lengths = inputs
        masks,indicator_masks = self.preprocess(measurements,artificial_mis=self.MIT_missing_rate)
        masks = masks if self.MIT and training else tf.cast(measurements, tf.float32)
        
        if training and self.MIT:
            X = values * masks
        else:
            X = values

        input_X_for_first = tf.concat([X, tf.cast(measurements, tf.float32)], 2) if self.input_with_mask else X                     # [n_batch, 300, x_dim*2 or x_dim]
        input_X_for_first = self.embedding_1(input_X_for_first)                                                              # [n_batch, 300, d_model]
        input_X_for_first += self.pos_encoding(input_X_for_first)
        enc_output = self.dropout(input_X_for_first)  

        if len(lengths.get_shape()) == 2:
            lengths = tf.squeeze(lengths, -1)

        mask = tf.sequence_mask(lengths, maxlen=tf.shape(times)[1]) # batch, max len
        enc_output = self.transformer1(enc_output,mask)

        X_tilde_1 = self.reduce_dim_z(enc_output)                                                           # [n_batch,300,x_dim]
        X_prime = masks * X + (1 - masks) * X_tilde_1                                                       # [n_batch,300,x_dim]
        input_X_for_second = tf.concat([X_prime, tf.cast(measurements, tf.float32)], 2) if self.input_with_mask else X_prime            # [n_batch, 300, x_dim*2 or x_dim]
        input_X_for_second = self.embedding_2(input_X_for_second)                                           # [n_batch, 300, d_model]
        enc_output += self.pos_encoding(input_X_for_second)
        enc_output = self.transformer2(enc_output,mask)        # attention mask diagnoal mask면 이거 만들어서 넣기
        
        X_tilde_2 = self.reduce_dim_gamma(tf.nn.relu(self.reduce_dim_beta(enc_output)))                     # [n_batch,300,x_dim]

        X_tilde_3 = 0.5 * (X_tilde_2 + X_tilde_1)
        X_c = masks * X + (1 - masks) * X_tilde_3  # replace non-missing part with original data - 이게 x_gen과 같음 shape = []

        ########## loss function 계산 #############
        def reconstruct_loss(x_tilde,x,masks):
            return tf.reduce_sum(abs(x_tilde-x)*masks)/(tf.reduce_sum(masks)+1e-09)

        L_ORT = (reconstruct_loss(X_tilde_1,X,masks)+
                 reconstruct_loss(X_tilde_2,X,masks)+
                 reconstruct_loss(X_tilde_3,X,masks))/3
        # tf.print(L_ORT)
        if self.MIT and training:
            L_MIT = reconstruct_loss(X_c,X,indicator_masks)
            loss=L_MIT
        else:
            loss=0
        padding_mask = mask

        y_prob = tf.ones(shape=[tf.shape(values)[0],1])
        loss += L_ORT
        prefix = "Train" if training else "Valid"

        aux={'x_impute':X_c
        ,            "metrics": {
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

    # def train_step(self, data):
    #     # from tensorflow.python.keras.engine import data_adapter
    #     # from tensorflow.python.eager import backprop

    #     # data = data_adapter.expand_1d(data)
    #     # x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
    #     # # Run forward pass.
    #     # with backprop.GradientTape() as tape:
    #     #     y_prob = self(x,output=y,training=True,return_aux=False)
    #     #     # Run backwards pass.
    #     # self.optimizer.minimize(self.loss, self.trainable_variables, tape=tape)

    #     # return_metrics = {"loss": self.loss}

    #     return return_metrics

    # def test_step(self, data):
    #     from tensorflow.python.keras.engine import data_adapter

    #     x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    #     y_prob = self(x,output=y,training=False,return_aux=False)
 
    #     return {"loss": self.loss}  # self.compute_metrics(x, y, y_pred, sample_weight)


    def get_config(self):
        return self._config

        # problem 1.
