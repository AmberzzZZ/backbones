from keras.layers import Input, Conv2D, add, Dropout, Dense, Lambda, GlobalAveragePooling2D, Layer
from WMSA import WindowMultiHeadAttention, FeedForwardNetwork, gelu
from LayerNormalization import LayerNormalization
from keras.models import Model
import keras.backend as K
import numpy as np
import tensorflow as tf
import math


def SwinTransformer(input_shape=(224,224,3), patch_size=4, emb_dim=96, ape=False, n_classes=1000,    # in/out hypers
                    num_layers=[2,2,6,2], num_heads=[3,6,12,24],                            # structual hypers
                    window_size=7, qkv_bias=True, qk_scale=None, mlp_ratio=4,               # swin-block hypers
                    attn_drop=0., ffn_drop=0., residual_drop=0.2):

    inpt = Input(input_shape)
    # assert input_size%7==0 and input_size%16==0, 'input_size can not be divided clean'

    # patch embedding
    x = Conv2D(emb_dim, patch_size, strides=patch_size, padding='same')(inpt)    # (b,h/4,w/4,C), autopad
    H, W = (math.ceil(input_shape[0]/patch_size), math.ceil(input_shape[1]/patch_size))
    x = LayerNormalization()(x)      # [b,H,W,D]

    # absolute positional embeddings
    if ape:
        pe = Lambda(lambda x: tf.truncated_normal((H,W,emb_dim), mean=0.0, stddev=.02))(x)  # (1,H,W,D)
        x = add([x, pe])   # (b,H,W,D)

    x = Dropout(ffn_drop)(x)

    # transformer stages
    n_stages = len(num_layers)
    dbr = np.linspace(0, residual_drop, num=sum(num_layers))    # drop block rate
    block_idx = 0
    for i in range(n_stages):
        merging = True if i<n_stages-1 else False
        # pad on the top
        pad_l = pad_t = 0
        pad_b, pad_r = (window_size - H % window_size) % window_size, (window_size - W % window_size) % window_size
        # x = Lambda(lambda x: tf.pad(x, [[0,0],[pad_t, pad_b], [pad_l, pad_r], [0,0]]))(x)
        x = Pad_HW(pad_t, pad_b, pad_l, pad_r)(x)
        H += pad_b
        W += pad_r
        # WSA+SWSA: 2 blocks
        x = basicStage(x, emb_dim, (H,W), num_layers[i]//2, num_heads[i], window_size, mlp_ratio, qkv_bias,
                         attn_drop=0., ffn_drop=0., residual_drop=dbr[sum(num_layers[:i]):sum(num_layers[:i+1])],
                         patch_merging=merging, idx=block_idx, stage=i)
        emb_dim *= 2
        block_idx += num_layers[i]//2
        if merging:
            H = (H+1) // 2
            W = (W+1) // 2
    x = LayerNormalization()(x)     # [b,H/32,W/32,8C]

    # head
    x = GlobalAveragePooling2D(data_format='channels_last')(x)    # (b,8C)

    if n_classes:
        x = Dense(n_classes, activation='softmax')(x)

    model = Model(inpt, x)

    return model


class Pad_HW(Layer):
    def __init__(self, pad_t, pad_b, pad_l, pad_r,**kwargs):
        super(Pad_HW, self).__init__(**kwargs)
        self.pad_t = pad_t
        self.pad_b = pad_b
        self.pad_l = pad_l
        self.pad_r = pad_r

    def call(self, x):
        return tf.pad(x, [[0,0],[self.pad_t, self.pad_b], [self.pad_l, self.pad_r], [0,0]])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]+self.pad_t+self.pad_b, input_shape[2]+self.pad_l+self.pad_r, input_shape[3])


# alternative [swin_block + patch_merging] for each stage
def basicStage(x, emb_dim, feature_shape, depth, n_heads, window_size, mlp_ratio=4, qkv_bias=True,
               attn_drop=0., ffn_drop=0., residual_drop=[], patch_merging=False, idx=None, stage=None):
    # assert depth==len(residual_drop)
    assert len(residual_drop) % 2 == 0
    # swin blocks
    for i in range(depth):
        x = SwinTransformerBlock(emb_dim, feature_shape, n_heads, window_size, mlp_ratio, qkv_bias,
                                 attn_drop, ffn_drop, residual_drop[i], idx=idx+i)(x)
    # downsampling
    if patch_merging:
        x = Lambda(PatchMerging, arguments={'feature_shape': feature_shape, 'emb_dim': emb_dim}, name='PatchMerging%d' % stage)(x)

    return x


class SwinTransformerBlock(Model):
    def __init__(self, emb_dim, feature_shape, n_heads, window_size, mlp_ratio=4, qkv_bias=True,
                 attn_drop=0., ffn_drop=0., residual_drop=0., idx=None, **kwargs):
        super(SwinTransformerBlock, self).__init__(name='STB_%d' % idx, **kwargs)
        self.emb_dim = emb_dim
        self.feature_shape = feature_shape
        self.window_size = window_size
        self.shift_size = window_size//2

        # W-MSA
        self.ln1 = LayerNormalization()
        self.wmsa = WindowMultiHeadAttention(emb_dim, n_heads, window_size, qkv_bias, attn_drop, ffn_drop)
        self.res_drop1 = Dropout(residual_drop, noise_shape=(None, 1, 1))

        self.ln2 = LayerNormalization()
        self.ffn = FeedForwardNetwork(emb_dim*mlp_ratio, emb_dim, activation=gelu, drop_rate=ffn_drop)
        self.res_drop2 = Dropout(residual_drop, noise_shape=(None, 1, 1))

        # SW-MSA
        self.ln3 = LayerNormalization()
        self.wmsa_s = WindowMultiHeadAttention(emb_dim, n_heads, window_size, qkv_bias, attn_drop, ffn_drop)
        self.res_drop3 = Dropout(residual_drop, noise_shape=(None, 1, 1))

        self.ln4 = LayerNormalization()
        self.ffn_s = FeedForwardNetwork(emb_dim*mlp_ratio, emb_dim, activation=gelu, drop_rate=ffn_drop)
        self.res_drop4 = Dropout(residual_drop, noise_shape=(None, 1, 1))

    def call(self, x):
        # input_shape: [b,H,W,D]
        # output_shape: [b,H,W,D]

        # WMSA block
        inpt = x
        x = self.ln1(x)
        x = WindowPartition(x, self.feature_shape, self.emb_dim, self.window_size)   # [b,nW,7x7,D]
        x = self.wmsa(x)
        x = WindowReverse(x, self.feature_shape, self.emb_dim, self.window_size)      # [b,H,W,D]
        x = self.res_drop1(x)
        x = x + inpt

        inpt = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.res_drop2(x)
        x = x + inpt

        # SWMSA block
        window_mask = getWindowMask(self.feature_shape, self.window_size, self.shift_size)  # (1,H,W,1)
        mask_windows = WindowPartition(window_mask, self.feature_shape, 1, self.window_size)   # (1,nW,7x7,1))
        mask_windows = tf.reshape(mask_windows, (-1,self.window_size*self.window_size))   # (nW,7x7)
        attn_mask = tf.expand_dims(mask_windows,axis=1) - tf.expand_dims(mask_windows,axis=2)   # (nW,7x7,7x7)
        attn_mask = tf.where(attn_mask==0, tf.zeros_like(attn_mask), -tf.ones_like(attn_mask)*100)
        attn_mask = tf.expand_dims(attn_mask, axis=1)    # (nW,1,7x7,7x7)

        inpt = x
        x = self.ln3(x)
        x = CyclicShift(x, self.feature_shape, self.emb_dim, self.shift_size)
        x = WindowPartition(x, self.feature_shape, self.emb_dim, self.window_size)   # [b,nW,local_L,D]
        x = self.wmsa_s(x, mask=attn_mask)
        x = WindowReverse(x, self.feature_shape, self.emb_dim, self.window_size)      # [b,T,D]
        x = CyclicShift(x, self.feature_shape, self.emb_dim, -self.shift_size)
        x = self.res_drop3(x)
        x = x + inpt

        inpt = x
        x = self.ln4(x)
        x = self.ffn_s(x)
        x = self.res_drop4(x)
        x = x + inpt

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def WindowPartition(x, feature_shape, n_channels, window_size):
    # x: [b,h,w,D]
    # output: [b,nW,local_L,D]

    b = tf.shape(x)[0]

    # split windows
    n_windows_h, n_windows_w = feature_shape[0]//window_size, feature_shape[1]//window_size
    x = tf.reshape(x, (b, n_windows_h, window_size, n_windows_w, window_size, n_channels))
    x = tf.transpose(x, (0,1,3,2,4,5))
    x = tf.reshape(x, (b, n_windows_h*n_windows_w, window_size*window_size, n_channels))   # [b,nW,Wh*Ww,D]
    return x


def WindowReverse(x, feature_shape, n_channels, window_size):
    # input: [b,nW,local_L,D]
    # output: [b,H,W,D]
    b = tf.shape(x)[0]
    n_windows_h, n_windows_w = feature_shape[0]//window_size, feature_shape[1]//window_size
    x = tf.reshape(x, (b, n_windows_h, n_windows_w, window_size, window_size, n_channels))
    x = tf.transpose(x, (0,1,3,2,4,5))
    x = tf.reshape(x, (b, n_windows_h*window_size, n_windows_w*window_size, n_channels))
    return x


def CyclicShift(x, feature_shape, n_channels, shift_size):
    # input: [b, H,W, D]
    # output: [b, H,W, D]
    x = tf.manip.roll(x, shift=[shift_size, shift_size], axis=[1,2])
    return x


def PatchMerging(x, feature_shape, emb_dim):
    # input: [b, H,W, D]
    # output: [b,H//2,W//2,D*2]

    h, w = feature_shape
    pad_h, pad_w = int(h%2==1), int(w%2==1)
    x = tf.pad(x, [[0,0],[0,pad_h],[0,pad_w],[0,0]], name=None)

    x0 = x[:,0::2,0::2,:]
    x1 = x[:,1::2,0::2,:]
    x2 = x[:,0::2,1::2,:]
    x3 = x[:,1::2,1::2,:]
    x = K.concatenate([x0,x1,x2,x3], axis=-1)  # [b,h/2,w/2,4c]
    x = Dense(2*emb_dim, use_bias=False)(x)

    return x


def getWindowMask(feature_shape, window_size, shift_size):

    img = np.zeros((1,feature_shape[0],feature_shape[1],1))
    h_slices = (slice(0,-window_size), slice(-window_size,-shift_size), slice(-shift_size,None))
    w_slices = (slice(0,-window_size), slice(-window_size,-shift_size), slice(-shift_size,None))

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img[:,h,w,:] = cnt
            cnt += 1

    img = tf.constant(img, dtype=tf.float32)
    return img


if __name__ == '__main__':

    model = SwinTransformer()
    model.summary()

    x = Input((224,224,3))
    y = model(x)
    print(y)






