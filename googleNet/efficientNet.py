from keras.layers import Input, Conv2D, BatchNormalization, Activation, DepthwiseConv2D, Multiply,\
                         GlobalAveragePooling2D, Reshape, Dropout, add, Dense
from keras.models import Model
import keras.backend as K


DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32,  'filters_out': 16,  'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16,  'filters_out': 24,  'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24,  'filters_out': 40,  'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40,  'filters_out': 80,  'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80,  'filters_out': 112, 'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192, 'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320, 'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
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


def EfficientNet(input_tensor=None, input_shape=(224,224,3), n_classes=1000,
                 drop_connect_rate=0., dropout_rate=0.):
    if input_tensor is not None:
        inpt = input_tensor
    else:
        inpt = Input(input_shape)

    # stem
    x = Conv_BN(inpt, 32, 3, strides=2)

    # blocks
    b = 0
    blocks = float(sum(args['repeats'] for args in DEFAULT_BLOCKS_ARGS))
    for block_arg in DEFAULT_BLOCKS_ARGS:
        for i in range(block_arg['repeats']):
            if i > 0:
                block_arg['filters_in'] = block_arg['filters_out']
                block_arg['strides'] = 1
            x = efficientBlock(x, drop_connect_rate*b/blocks, **block_arg)
            b += 1

    x = Conv_BN(x, 1280, 1, strides=1)

    # top
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = Dense(n_classes, activation='softmax', kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)

    model = Model(inpt, x)

    return model


def efficientBlock(x, drop_rate=0., kernel_size=3, repeats=1, filters_in=32, filters_out=16, expand_ratio=1, id_skip=True, strides=1, se_ratio=0.25):
    inpt = x

    # PW-expand
    if expand_ratio > 1:
        n_filters = filters_in * expand_ratio
        x = Conv_BN(x, n_filters, kernel_size=1, strides=1)

    # DW conv
    x = DW_Conv_BN(x, kernel_size=3, strides=strides)

    # SE-block
    if se_ratio:
        x = se_block(x, int(filters_in*se_ratio))

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
    x = Conv2D(dense_dim, 1, strides=1, padding='same',
               activation=swish, kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    # excite
    x = Conv2D(in_channels, 1, strides=1, padding='same',
               activation='sigmoid', kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    # reweight
    x = Multiply()([inpt, x])
    return x


def swish(x):
    return x * K.sigmoid(x)


def Conv_BN(x, n_filters, kernel_size=3, strides=1, kernel_initializer=CONV_KERNEL_INITIALIZER, activation=swish):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x


def DW_Conv_BN(x, kernel_size=3, strides=1, kernel_initializer=CONV_KERNEL_INITIALIZER, activation=swish):
    x = DepthwiseConv2D(kernel_size, strides=strides, padding='same', depth_multiplier=1, depthwise_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x


if __name__ == '__main__':

    model = EfficientNet(input_shape=(224,224,3), n_classes=1000)
    model.summary()




