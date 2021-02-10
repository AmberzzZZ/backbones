# resnet 50 & 101
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, ReLU, add, GlobalAveragePooling2D, \
                         Reshape, Dense, multiply, Softmax, Lambda, Permute
from keras.models import Model
import keras.backend as K
import tensorflow as tf


# configurations for 4 stages
n_blocks = {50: [3,4,6,3], 101: [3,4,23,3]}
n_filters = [256, 512, 1024, 2048]
n_strides = [1,2,2,2]


def gcnet(input_shape=(224,224,3), depth=50, pooling=False):
    inpt = Input(input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 7, strides=2, activation='leaky')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[depth]
    for i in range(len(num_blocks)):
        for j in range(num_blocks[i]):
            strides = n_strides[i] if j==0 else 1
            x = res_block(x, n_filters[i], strides)

    if pooling:
        x = GlobalAveragePooling2D()(x)

    # model
    model = Model(inpt, x)

    return model


def res_block(x, n_filters, strides, gc=True):
    inpt = x
    # residual
    x = Conv_BN(x, n_filters//4, 1, strides=strides, activation='relu')
    x = Conv_BN(x, n_filters//4, 3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    if gc:
        x = gc_block(x)
    # shortcut
    if strides!=1 or inpt._keras_shape[-1]!=n_filters:
        inpt = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
    x = add([inpt, x])
    x = ReLU()(x)
    return x


def gc_block(x, pooling_type='att', se_ratio=16):
    inpt = x
    h, w, c = K.int_shape(x)[1:]
    # global context module
    if pooling_type=='avg':   # se-block
        x = GlobalAveragePooling2D()(inpt)
    else:    # context module, reweighting attention map
        x = Conv2D(1, kernel_size=1, strides=1, padding='same')(inpt)   # [b,h,w,1]
        x = Softmax(axis=-1)(x)
        x = Reshape((h*w,1))(x)   # [b,hw,1]

        x1 = Permute((3,1,2))(inpt)
        x1 = Reshape((c,-1))(x1)    # [b,c,hw]

        x = Lambda(lambda x: tf.matmul(x[0], x[1]))([x1, x])   # [b,c]
        x = Reshape((c,1,1))(x)   # [b,c,1,1]
        x = Permute((2,3,1))(x)   # [b,1,1,c]

    # se transform module
    sq_filters = c // se_ratio
    x = Conv_BN(x, sq_filters, kernel_size=1, strides=1, activation='relu')
    x = Conv2D(c, kernel_size=1, strides=1, padding='same')(x)

    # add
    x = add([inpt, x])

    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = gcnet(input_shape=(224,224,3), depth=50)
    model.summary()




