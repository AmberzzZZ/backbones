# resnext, 50 & 101, use self defined groupConv Layer
from keras.layers import Input, MaxPooling2D, LeakyReLU, add, BatchNormalization, Lambda, concatenate, Conv2D, GlobalAveragePooling2D, Softmax, Reshape
from keras.models import Model
import keras.backend as K
import tensorflow as tf


n_blocks = [3,3,5,2]
g_filters = [128, 256, 512, 1024]          # group filters for resnext with C=32
n_filters = [256, 512, 1024, 2048]         # output filters


def csp_r50(input_shape=(256,256,3), n_classes=1000):
    inpt = Input(shape=input_shape)

    # stem
    x = Conv_BN(inpt, 64, 7, strides=2, activation='leaky')
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    # csp block1 : 64x64x64
    csp_skip = Conv_BN(x, 128, 1, strides=1, activation=None)
    # res entry
    x = Conv_BN(x, 64, 1, strides=1, activation='leaky')
    # res blocks
    x = res_block(x, 128, 3, strides=1)
    x = res_block(x, 128, 3, strides=1)
    x = res_block(x, 128, 3, strides=1)
    # inside transition
    x = Conv_BN(x, 128, 1, strides=1, activation='leaky')
    # concat
    x = concatenate([csp_skip, x], axis=-1)
    # outside transition
    x = Conv_BN(x, 128, 1, strides=1, activation='leaky')

    # downsamp
    x = Conv_BN(x, 128, 3, strides=2, activation='leaky')

    # csp block2: 32x32x128
    csp_skip = Conv_BN(x, 256, 1, strides=1, activation=None)
    # res entry
    x = Conv_BN(x, 256, 1, strides=1, activation=None)
    # res blocks
    x = res_block(x, 256, 3, strides=1)
    x = res_block(x, 256, 3, strides=1)
    x = res_block(x, 256, 3, strides=1)
    # inside transition
    x = Conv_BN(x, 256, 1, strides=1, activation='leaky')
    # concat
    x = concatenate([csp_skip, x], axis=-1)
    # outside transition
    x = Conv_BN(x, 256, 1, strides=1, activation='leaky')

    # downsamp
    x = Conv_BN(x, 256, 3, strides=2, activation='leaky')

    # csp block3: 16x16x256
    csp_skip = Conv_BN(x, 512, 1, strides=1, activation=None)
    # res entry
    x = Conv_BN(x, 512, 1, strides=1, activation=None)
    # res blocks
    x = res_block(x, 512, 3, strides=1)
    x = res_block(x, 512, 3, strides=1)
    x = res_block(x, 512, 3, strides=1)
    x = res_block(x, 512, 3, strides=1)
    x = res_block(x, 512, 3, strides=1)
    # inside transition
    x = Conv_BN(x, 512, 1, strides=1, activation='leaky')
    # concat
    x = concatenate([csp_skip, x], axis=-1)
    # outside transition
    x = Conv_BN(x, 512, 1, strides=1, activation='leaky')

    # downsamp
    x = Conv_BN(x, 512, 3, strides=2, activation='leaky')

    # csp block4: 8x8x512
    csp_skip = Conv_BN(x, 1024, 1, strides=1, activation=None)
    # res entry
    x = Conv_BN(x, 1024, 1, strides=1, activation=None)
    # res blocks
    x = res_block(x, 1024, 3, strides=1)
    x = res_block(x, 1024, 3, strides=1)
    # inside transition
    x = Conv_BN(x, 1024, 1, strides=1, activation='leaky')
    # concat
    x = concatenate([csp_skip, x], axis=-1)
    # outside transition
    x = Conv_BN(x, 1024, 1, strides=1, activation='leaky')

    # head
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,1024))(x)
    x = Conv2D(n_classes, 1, strides=1, padding='same', activation=None, use_bias=True)(x)
    x = Softmax()(x)

    # model
    model = Model(inpt, x)

    return model


def res_block(x, n_filters, kernel_size, strides):
    skip = x
    if K.int_shape(skip)[-1] < n_filters:
        gap = n_filters - K.int_shape(skip)[-1]
        skip = Lambda(lambda x: tf.pad(x, [[0,0],[0,0],[0,0],[0,gap]]))(skip)
    x = Conv_BN(x, n_filters//2, 1, strides=1, activation='leaky')
    x = Conv_BN(x, n_filters//2, 3, strides=1, activation='leaky')
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    x = add([skip, x])
    x = LeakyReLU()(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = LeakyReLU()(x)
    return x


if __name__ == '__main__':

    model = csp_r50(input_shape=(256,256,3), n_classes=1000)
    model.load_weights('csp_r50.h5', by_name=True)




