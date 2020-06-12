# resnest, 50 & 101
from keras.layers import Input, MaxPooling2D, ReLU, add, concatenate, Lambda, AveragePooling2D, Conv2D,  \
                         BatchNormalization, GlobalAveragePooling2D, Dropout, Dense, Reshape, Softmax, multiply
from keras.models import Model
import tensorflow as tf
from groupConv import GroupConv2D


n_blocks = {50: [3,4,6,3], 101: [3,4,23,3], 200: [3,24,36,3], 269: [3,30,48,8]}
g_filters = [64, 128, 256, 512]          # bottleneck d for resnet, group filters for resneSt for C=1,r=2
# g_filters = [128, 256, 512, 1024]        # group filters for resnext with C=32
n_filters = [256, 512, 1024, 2048]         # output filters


def resneSt(input_shape=(224,224,3), depth=50, r=2, C=1, n_classes=1000,
            bottleneck_width=64, deep_stem=True, stem_width=32, drop_out=0.,
            avg_down=True):
    inpt = Input(input_shape)

    # stem: conv+bn+relu+pool, 7x7 / 3x3+3x3+3x3
    if deep_stem:
        x = Conv_BN(inpt, stem_width, 3, strides=2, activation='relu')
        x = Conv_BN(x, stem_width, 3, strides=1, activation='relu')
        x = Conv_BN(x, stem_width*2, 3, strides=1, activation='relu')
    else:
        x = Conv_BN(inpt, 64, 7, strides=2, activation='relu')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[depth]
    for lv in range(len(num_blocks)):
        for i in range(num_blocks[lv]):
            strides = 2 if (i==0 and lv!=0) else 1
            x = resnest_block(x, g_filters[lv], n_filters[lv], strides, r, C, avg_down)

    # # head
    # x = GlobalAveragePooling2D()(x)
    # x = Dropout(drop_out)(x) if drop_out>0 else x
    # x = Dense(n_classes)(x)

    # model
    model = Model(inpt, x)

    return model


def resnest_block(x, g_filters, filters, strides=1, r=2, C=1, avg_down=True):
    inpt = x
    # res path: group blocks 1x1 3x3
    # downsamp
    if avg_down:
        x = AveragePooling2D(padding='same')(x)
        x = Conv_BN(x, g_filters, kernel_size=1, strides=1, activation='relu')
    else:
        x = Conv_BN(x, g_filters, kernel_size=1, strides=strides, activation='relu')

    x = Conv_BN(x, g_filters, kernel_size=3, strides=1, group=r*C, activation='relu')

    # split-attention module & concat
    if r==1:
        # 3x3 conv
        x = Conv_BN(x, filters, kernel_size=1, strides=1, activation=None)
    else:
        # split attention
        if C>1:
            x = Lambda(lambda x: tf.split(x,C))(x)
            x = [split_at(i, g_filters, r=r) for i in x]
            x = concatenate(x, axis=-1)
        else:
            x = split_at(x, g_filters, r=r)

        # 1x1 conv
        x = Conv_BN(x, filters, kernel_size=1, strides=1, activation=None)

    # id path
    if strides>1 or inpt._keras_shape[-1]!=filters:
        if avg_down:
            inpt = AveragePooling2D(padding='same')(inpt)
            inpt = Conv_BN(inpt, filters, 1, 1, activation=None)
        else:
            inpt = Conv_BN(inpt, filters, 1, strides, activation=None)

    x = add([inpt, x])
    x = ReLU()(x)

    return x


def split_at(x, filters, kernel_size=3, r=1):
    inpt = x
    # add
    x = Lambda(lambda x: tf.split(x,r,axis=-1))(x)
    x = add(x)      # c/r
    # gap
    x = GlobalAveragePooling2D()(x)
    # dense1: squeeze
    x = Dense(filters//4)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # dense2: expand
    x = Dense(filters)(x)     # c
    # r-softmax
    x = Reshape((r,filters//r))(x)
    x = Softmax(axis=-1)(x)
    x = Reshape((1,1,filters))(x)
    x = Lambda(lambda x: tf.split(x,r,axis=-1))(x)
    # fusion
    inpt = Lambda(lambda x: tf.split(x,r,axis=-1))(inpt)
    x = [multiply([feature, weight]) for feature,weight in zip(inpt, x)]
    x = add(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, group=1, activation=None):
    x = GroupConv2D(n_filters, kernel_size, strides=strides, padding='same', group=group)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    # model = resneSt(r=1,C=1,deep_stem=False,avg_down=False)     # resnet
    # model = resneSt(r=1,C=32,deep_stem=False,avg_down=False)     # resnext
    model = resneSt(r=2,C=1,deep_stem=False,avg_down=False)     # resneSt
    model.summary()

    # x = Input((28,28,16))
    # y = split_at(x, 16, kernel_size=3, r=2)
    # model = Model(x, y)
    # model.summary()
