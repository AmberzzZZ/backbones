from keras.layers import Input, Conv2D, BatchNormalization, Activation, DepthwiseConv2D, Multiply,\
                         GlobalAveragePooling2D, Reshape, Dropout, add, Dense
from keras.models import Model
import keras.backend as K
import math
import copy


DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def EfficientNet(input_shape, width_coefficient, depth_coefficient, dropout_rate=0.2,
                 n_classes=1000, drop_connect_rate=0.2, blocks_args=DEFAULT_BLOCKS_ARGS,
                 depth_divisor=8):

    blocks_args = copy.deepcopy(blocks_args)

    if isinstance(input_shape, tuple):
        inpt = Input(input_shape)
    else:
        inpt = Input((input_shape,input_shape,3))

    # stem
    x = Conv_BN(inpt, round_filters(32, width_coefficient), 3, strides=2)

    # blocks
    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    # for each block group
    for block_arg in blocks_args:
        # for each mbconv block
        block_arg['repeats'] = round_repeats(block_arg['repeats'], depth_coefficient)
        block_arg['filters_in'] = round_filters(block_arg['filters_in'], width_coefficient)
        block_arg['filters_out'] = round_filters(block_arg['filters_out'], width_coefficient)
        for i in range(block_arg['repeats']):
            if i > 0:
                block_arg['filters_in'] = block_arg['filters_out']
                block_arg['strides'] = 1
            x = efficientBlock(x, drop_connect_rate*b/blocks, **block_arg)
            b += 1

    # top
    x = Conv_BN(x, round_filters(1280, width_coefficient), 1, strides=1)
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = Dense(n_classes, activation='softmax', kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)

    model = Model(inpt, x)

    return model


def efficientBlock(x, drop_rate=0., kernel_size=3, repeats=1, filters_in=32, filters_out=16,
                   expand_ratio=1, id_skip=True, strides=1, se_ratio=0.25):
    inpt = x

    # PW-expand
    if expand_ratio > 1:
        n_filters = filters_in * expand_ratio
        x = Conv_BN(x, n_filters, kernel_size=1, strides=1)

    # DW conv
    x = DW_Conv_BN(x, kernel_size=kernel_size, strides=strides)

    # SE-block
    if se_ratio:
        filters_se = max(1, int(filters_in*se_ratio))
        x = se_block(x, filters_se)

    # PW-project
    x = Conv_BN(x, filters_out, kernel_size=1, strides=1, activation=None)

    # residual
    if id_skip is True and strides==1 and filters_in==filters_out:
        if drop_rate > 0:
            x = Dropout(drop_rate, noise_shape=(None, 1, 1, 1))(x)
        x = add([x, inpt])

    return x


def se_block(x, dense_dim):
    inpt = x
    in_channels = x._keras_shape[-1]
    # squeeze
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,in_channels))(x)
    # reduce
    x = Conv2D(dense_dim, 1, strides=1, padding='same', use_bias=False,
               activation=swish, kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    # excite
    x = Conv2D(in_channels, 1, strides=1, padding='same', use_bias=False,
               activation='sigmoid', kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    # reweight
    x = Multiply()([inpt, x])
    return x


def swish(x):
    return x * K.sigmoid(x)


def Conv_BN(x, n_filters, kernel_size=3, strides=1, padding='same', activation=swish,
            kernel_initializer=CONV_KERNEL_INITIALIZER):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding=padding, use_bias=False, kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x


def DW_Conv_BN(x, kernel_size=3, strides=1, padding='same', activation=swish,
               kernel_initializer=CONV_KERNEL_INITIALIZER):
    x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False, depthwise_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x


def round_filters(filters, width_coefficient, divisor=8):
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))


########### effNets family ########
def EfficientNetB0():
    return EfficientNet(224, 1.0, 1.0, 0.2)


def EfficientNetB1():
    return EfficientNet(240, 1.0, 1.1, 0.2)


def EfficientNetB2():
    return EfficientNet(260, 1.1, 1.2, 0.3)


def EfficientNetB3():
    return EfficientNet(300, 1.2, 1.4, 0.3)


def EfficientNetB4():
    return EfficientNet(380, 1.4, 1.8, 0.4)


def EfficientNetB5():
    return EfficientNet(456, 1.6, 2.2, 0.4)


def EfficientNetB6():
    return EfficientNet(528, 1.8, 2.6, 0.5)


def EfficientNetB7():
    return EfficientNet(600, 2.0, 3.1, 0.5)


if __name__ == '__main__':

    model = EfficientNet(512, 1.8, 2.6, 0.5)
    model.load_weights("/Users/amber/Downloads/efficientnet-b6_noisy-student_notop.h5")
    model.summary()

    # newmodel = Model(model.input, model.get_layer(index=-4).output)
    # newmodel.summary()
    # newmodel.save_weights('eff_b4_notop.h5')
    # print(newmodel.input, newmodel.get_layer(index=1).output)





