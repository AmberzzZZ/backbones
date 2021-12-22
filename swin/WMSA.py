from keras.layers import Input, Dense, Dropout, Activation, Layer
from keras.models import Model
from keras.activations import relu
import tensorflow as tf
import keras.backend as K
import math
import numpy as np


def gelu(x, approx=False):
    if approx:
        return 0.5 * x * (1 + K.tanh(K.sqrt(K.constant(2./math.pi)) * (x + 0.044715 * K.pow(x, 3))))
    else:
        return 0.5 * x * (1. + tf.math.erf(x / K.sqrt(K.constant(2.))))


def bias_init(shape, dtype=None):
    return tf.truncated_normal(shape, mean=0.0, stddev=.02, dtype=tf.float32)


# Window based multi-head self attention with relative position bias(qkv-bias)
class WindowMultiHeadAttention(Model):
    def __init__(self, emb_dim, num_heads, window_size, qkv_bias=True, attn_drop=0., ffn_drop=0.,
                 **kwargs):
        super(WindowMultiHeadAttention, self).__init__(**kwargs)
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        assert emb_dim%num_heads==0, 'not exact division in WMSA'
        self.head_dim = emb_dim // num_heads
        self.window_size = window_size

        self.QKV = Dense(3*emb_dim, use_bias=True, kernel_initializer=bias_init, bias_initializer='zeros')
        self.dense = Dense(emb_dim, kernel_initializer=bias_init, bias_initializer='zeros')
        self.msa_drop = Dropout(attn_drop)
        self.mlp_drop = Dropout(ffn_drop)

        h = w = window_size
        relative_bias_shape = ((2*h-1)*(2*w-1),num_heads)
        # tf.truncated_normal(((2*h-1)*(2*w-1),num_heads), mean=0.0, stddev=.02)   # [2h-1*2w-1,n_heads]
        # self.relative_position_bias = tf.Variable(initial_value, trainable=True)  # trainable, only for eager mode
        self.relative_position_bias = RelativePositionBias(relative_bias_shape, bias_init, trainable=True)
        self.relative_position_index = get_relative_dis_mat(h,w)     # constant, relative indices

    def call(self, x, mask=None):
        # window-based self-attention
        # input_shape: [b,nW,L,D], shared among n_windows&batches
        # mask_shape: [n_windows, L, L], connections among each window
        b = tf.shape(x)[0]
        nW = K.int_shape(x)[1]
        L = K.int_shape(x)[2]   # h*w
        assert L==self.window_size*self.window_size, 'dim not match'
        x = tf.reshape(self.QKV(x), (b, nW, L, 3, self.num_heads, self.head_dim))

        # qkv shape: [b, nW, num_heads, L, head_dim]
        query = tf.transpose(x[:,:,:,0], (0,1,3,2,4))
        key = tf.transpose(x[:,:,:,1], (0,1,3,2,4))
        value = tf.transpose(x[:,:,:,2], (0,1,3,2,4))

        # shape: (b, nW, num_heads, L, L)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        score = matmul_qk / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        # relative position bias: (L,L,num_heads)
        relative_position_bias = self.relative_position_bias(x)
        relative_position_bias = tf.gather(relative_position_bias, self.relative_position_index)   # (L,L,n_heads)
        relative_position_bias = tf.transpose(tf.reshape(relative_position_bias, (L, L, -1)), (2,0,1))  # (n_heads,L,L)

        # shape: (b, nW, num_heads, L, L)
        score += relative_position_bias

        # shape: (b, nW, num_heads, L, L)
        if mask is not None:
            score += mask     # 0 pos & -100 for neg

        # softmax & dropout
        alpha = tf.nn.softmax(score)
        alpha = self.msa_drop(alpha)

        # mul value & merge
        context = tf.matmul(alpha, value)  # shape: (b, nW, num_heads, L, D)
        context = tf.transpose(context, [0, 1, 3, 2, 4])  # shape: (b, nW, L, num_heads, head_dim)
        context = tf.reshape(context, (b, nW, L, self.emb_dim))   # shape: (b, nW, L, D)
        output = self.dense(context)
        output = self.mlp_drop(output)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape


def get_relative_dis_mat(h,w):
    # abs coords
    coord_w, coord_h = np.meshgrid(np.arange(w), np.arange(h))
    index = np.stack([coord_h,coord_w], axis=-1).reshape((-1,2))
    # relative coords
    dis = index[:,None,:] - index[None,:,:]    # [hw,hw,2]
    # shift
    dis[:,:,0] += h-1
    dis[:,:,1] += w-1
    # fuse by digit
    dis = dis[...,0]*(2*w-1) + dis[...,1]   # [hw,hw]
    dis = tf.constant(dis)
    return dis


class RelativePositionBias(Layer):
    def __init__(self, shape, initializer, trainable=True, **kargs):
        super(RelativePositionBias, self).__init__(**kargs)
        self.a = self.add_weight(shape=shape, initializer=initializer, trainable=trainable,
                                 name='relative_position_bias')

    def call(self, x):
        return self.a

    def compute_output_shape(self, input_shape):
        return self.a.shape


# FFN layer
class FeedForwardNetwork(Model):
    def __init__(self, dff_size, model_size, activation=relu, drop_rate=0.):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = Dense(dff_size, activation=activation, kernel_initializer=bias_init, bias_initializer='zeros')   # relu/gelu
        self.dense2 = Dense(model_size, kernel_initializer=bias_init, bias_initializer='zeros')
        if drop_rate:
            self.drop1 = Dropout(drop_rate)
            self.drop2 = Dropout(drop_rate)
        self.act = Activation(activation)
        self.model_size = model_size
        self.drop_rate = drop_rate

    def call(self, x):
        x = self.dense1(x)
        x = self.act(x)
        if self.drop_rate:
            x = self.drop1(x)
        x = self.dense2(x)
        if self.drop_rate:
            x = self.drop2(x)
        return x

    def compute_output_shape(self, input_shape):
        if len(input_shape)==4:
            B, H, W, _ = input_shape
            return (B,H,W,self.model_size)
        else:
            B, L, _ = input_shape
            return (B,L,self.model_size)


if __name__ == '__main__':

    # test MSA & FFN layer
    x = Input((16, 49, 10))   # [b,nW,L,D]
    mask = Input((16,1,49,49))    # [nW,1,L,L]
    y = WindowMultiHeadAttention(32, 4, 7)(inputs=x, mask=mask)
    # y = WindowMultiHeadAttention(10, 2)(inputs=[y,y,y], mask=mask)
    # y = FeedForwardNetwork(16, 10)(y)

    model = Model([x,mask],y)
    model.summary()
    for l in model.layers:
        try:
            for subl in l.layers:
                print(subl.name)
                print([i.shape for i in subl.get_weights()])
        except:
            continue
    # dense1, qkv, with bias, (C,3C),(3C)
    # relative bias, 
    # dense2, fuse_heads, with_bias, (C,C),(C,)

