# resnest, 50 & 101
from keras.layers import Input, MaxPooling2D, ReLU, add, concatenate, Lambda, AveragePooling2D, Conv2D, BatchNormalization
from keras.models import Model
import tensorflow as tf
from resnet import Conv_BN


n_blocks = {50: [3,4,6,3], 101: [3,4,23,3], 200: [3,24,36,3], 269: [3,30,48,8]}
n_filters = [256, 512, 1024, 2048]


def resneSt(input_shape=(224,224,3), depth=50, r=2, C=1):
    inpt = Input(input_shape)

    # stem: conv+bn+relu+pool, 7x7 / 3x3+3x3+3x3
    x = Conv_BN(inpt, 64, 7, strides=2, activation='relu')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[depth]
    x = resnest_block(x, n_filters[0], num_blocks[0])
    x = resnest_block(x, n_filters[1], num_blocks[1], strides=2)
    x = resnest_block(x, n_filters[2], num_blocks[2], strides=2)
    x = resnest_block(x, n_filters[3], num_blocks[3], strides=2)

    # model
    model = Model(inpt, x)

    return model


def resnest_block(x, filters, strides=1, avg_down=True, r=2, C=1):
    inpt = x
    # res path
    x = 
    # group blocks

    
    # id path
    if strides!=1 or x._keras_shape[-1]!=filters:
        if avg_down:
            inpt = AveragePooling2D(pool_size=strides, strides=strides, padding='same')(inpt)
            inpt = Conv2D(filters, 1, strides=1, padding='same')(inpt)
        else:
            inpt = Conv2D(filters, 1, strides=strides, padding='same')(inpt)
        inpt = BatchNormalization()(inpt)





def splat_block(x, filters=64, kernel_size=3, stride=1, dilation=1, groups=1, radix=0):


