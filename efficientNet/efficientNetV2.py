from keras.layers import Input, Conv2D, BatchNormalization, Activation, DepthwiseConv2D, Multiply,\
                         GlobalAveragePooling2D, Reshape, Dropout, add, Dense
from keras.models import Model
import keras.backend as K
import math
import copy
from effv2_cfg import V2S_BLOCKS_ARGS, V2M_BLOCKS_ARGS, V2L_BLOCKS_ARGS


DEFAULT_BLOCKS_ARGS = [   # V2-S
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 24, 'filters_out': 24,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.},
    {'kernel_size': 3, 'repeats': 4, 'filters_in': 24, 'filters_out': 48,
     'expand_ratio': 4, 'id_skip': True, 'strides': 2, 'se_ratio': 0.},
    {'kernel_size': 3, 'repeats': 4, 'filters_in': 48, 'filters_out': 64,
     'expand_ratio': 4, 'id_skip': True, 'strides': 2, 'se_ratio': 0.},
    {'kernel_size': 3, 'repeats': 6, 'filters_in': 64, 'filters_out': 128,
     'expand_ratio': 4, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 9, 'filters_in': 128, 'filters_out': 160,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 15, 'filters_in': 160, 'filters_out': 256,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
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


def EfficientNetV2(input_size, width_coefficient=1., depth_coefficient=1., dropout_rate=0.1, n_classes=1000,
                   drop_connect_rate=0.2, blocks_args=DEFAULT_BLOCKS_ARGS,
                   depth_divisor=8):

    blocks_args = copy.deepcopy(blocks_args)

    inpt = Input((input_size,input_size,3))

    # stem
    x = Conv_BN(inpt, round_filters(24, width_coefficient), 3, strides=2)

    # blocks
    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    # for each block group
    for block_arg in blocks_args[:]:
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
    x = Conv_BN(x, round_filters(1280, width_coefficient), 1, strides=1)   # 1792 in paper

    # x = GlobalAveragePooling2D()(x)
    # if dropout_rate > 0:
    #     x = Dropout(dropout_rate)(x)
    # x = Dense(n_classes, activation='softmax', kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)

    model = Model(inpt, x)

    return model


def efficientBlock(x, drop_rate=0., kernel_size=3, repeats=1, filters_in=32, filters_out=16,
                   expand_ratio=1, id_skip=True, strides=1, se_ratio=0.25):
    inpt = x

    if se_ratio:      # MBConv
        # PW-expand
        if expand_ratio > 1:
            n_filters = int(filters_in * expand_ratio)
            x = Conv_BN(x, n_filters, kernel_size=1, strides=1)

        # DW conv
        x = DW_Conv_BN(x, kernel_size=kernel_size, strides=strides)

        # SE-block
        filters_se = max(1, int(filters_in*se_ratio))
        x = se_block(x, filters_se)

    else:       # Fused-MBConv
        # 3x3 conv
        n_filters = int(filters_in * expand_ratio)
        x = Conv_BN(x, n_filters, kernel_size=kernel_size, strides=strides)

    # PW-project
    if expand_ratio>1:
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
    x = Conv2D(dense_dim, 1, strides=1, padding='same', use_bias=True, activation=swish)(x)
    # excite
    x = Conv2D(in_channels, 1, strides=1, padding='same', use_bias=True, activation='sigmoid')(x)
    # reweight
    x = Multiply()([inpt, x])
    return x


def swish(x):
    return x * K.sigmoid(x)


def Conv_BN(x, n_filters, kernel_size=3, strides=1, padding='same', activation=swish,
            kernel_initializer=CONV_KERNEL_INITIALIZER):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding=padding, use_bias=False,
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.9)(x)
    if activation:
        x = Activation(activation)(x)
    return x


def DW_Conv_BN(x, kernel_size=3, strides=1, padding='same', activation=swish,
               kernel_initializer=CONV_KERNEL_INITIALIZER):
    x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False,
                        depthwise_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.9)(x)
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


def EffV2S(input_size, n_classes):
    # Image Size schedule: 128-300
    # dropout schedule: 0.1-0.3
    # randaug: 10
    # mixup alpha: 0
    return EfficientNetV2(input_size, dropout_rate=0.3, n_classes=n_classes,
                          blocks_args=V2S_BLOCKS_ARGS)


def EffV2M(input_size, n_classes):
    # Image Size schedule: 128-380
    # dropout schedule: 0.1-0.4
    # randaug: 15
    # mixup alpha: 0.2
    return EfficientNetV2(input_size, dropout_rate=0.4, n_classes=n_classes,
                          blocks_args=V2M_BLOCKS_ARGS)


def EffV2L(input_size, n_classes):
    # Image Size schedule: 128-380
    # dropout schedule: 0.1-0.5
    # randaug: 20
    # mixup alpha: 0.5
    return EfficientNetV2(input_size, dropout_rate=0.5, n_classes=n_classes,
                          blocks_args=V2L_BLOCKS_ARGS)


if __name__ == '__main__':

    # model = EfficientNetV2(input_size=128, n_classes=30, dropout_rate=0.1)
    # model = EffV2M(input_size=380, n_classes=54)
    # model = EffV2S(input_size=300, n_classes=54)
    model = EffV2M(input_size=380, n_classes=1000)
    # model.load_weights("/Users/amber/Downloads/Misc/efficientnetv2-s-21k.h5")
    model.summary()





