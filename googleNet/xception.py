from keras.layers import Conv2D, BatchNormalization, ReLU, add, Input, DepthwiseConv2D, MaxPooling2D,  \
                         GlobalAveragePooling2D, SeparableConv2D
from keras.models import Model


def Xception(input_tensor=None, input_shape=(299,299,3), n_classes=1000):
    if input_tensor is not None:
        inpt = input_tensor
    else:
        inpt = Input(input_shape)

    # entry flow
    x = Conv_BN(inpt, 32, kernel_size=3, strides=2)
    x = Conv_BN(x, 64, kernel_size=3, strides=1)
    x = sep_conv_block(x, filters=128, strides=2)
    x = sep_conv_block(x, filters=256, strides=2)
    x = sep_conv_block(x, filters=728, strides=2)
    # middle flow
    for i in range(8):
        x = sep_id_block(x, filters=728)
    # exit flow
    x = sep_conv_block(x, filters=[728, 1024], strides=2)
    x = Sep_Conv_BN(x, filters=1536, strides=1, activation=True, deeplab=False)
    x = Sep_Conv_BN(x, filters=2048, strides=1, activation=True, deeplab=False)
    # head
    x = GlobalAveragePooling2D()(x)

    model = Model(inpt, x)

    return model


def Xception_deeplab(input_tensor=None, input_shape=(299,299,3), n_classes=1000):
    if input_tensor is not None:
        inpt = input_tensor
    else:
        inpt = Input(input_shape)

    # entry flow
    x = Conv_BN(inpt, 32, kernel_size=3, strides=2)
    x = Conv_BN(x, 64, kernel_size=3, strides=1)
    x = sep_conv_block_deeplab(x, filters=128, strides=2)
    x = sep_conv_block_deeplab(x, filters=256, strides=2)
    x = sep_conv_block_deeplab(x, filters=728, strides=2)
    # middle flow
    for i in range(16):
        x = sep_id_block(x, filters=728)
    # exit flow
    x = sep_conv_block_deeplab(x, filters=(728,1024,1024), strides=2)
    x = Sep_Conv_BN(x, filters=1536, strides=1)
    x = Sep_Conv_BN(x, filters=1536, strides=1)
    x = Sep_Conv_BN(x, filters=2048, strides=1)

    model = Model(inpt, x)

    return model


def Conv_BN(x, filters, kernel_size, strides, activation=True):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


# with conv_bn in id path, maxpooling in the res path
def sep_conv_block(x, filters, strides):
    if isinstance(filters, int):
        inpt = x
        inpt = Conv_BN(inpt, filters, kernel_size=1, strides=strides, activation=False)

        x = Sep_Conv_BN(x, filters, strides=1, activation=True, deeplab=False)
        x = Sep_Conv_BN(x, filters, strides=1, activation=False, deeplab=False)
        x = MaxPooling2D(pool_size=3, strides=strides, padding='same')(x)
    else:       # n_filter differs for each layer in the block
        inpt = x
        inpt = Conv_BN(inpt, filters[-1], kernel_size=1, strides=strides, activation=False)

        for idx, f in enumerate(filters):
            activation = True if idx < len(filters)-1 else False
            x = Sep_Conv_BN(x, filters=f, strides=1, activation=activation, deeplab=False)
        x = MaxPooling2D(pool_size=3, strides=strides, padding='same')(x)

    x = add([inpt, x])
    x = ReLU()(x)

    return x


# with conv_bn in id path, may have stride2 in last conv
def sep_conv_block_deeplab(x, filters, strides):
    if isinstance(filters, int):
        inpt = x
        inpt = Conv_BN(inpt, filters, kernel_size=1, strides=strides, activation=False)

        x = Sep_Conv_BN(x, filters, strides=1, activation=True, deeplab=True)
        x = Sep_Conv_BN(x, filters, strides=1, activation=True, deeplab=True)
        x = Sep_Conv_BN(x, filters, strides=strides, activation=False, deeplab=True)
    else:       # n_filter differs for each layer in the block
        inpt = x
        inpt = Conv_BN(inpt, filters[-1], kernel_size=1, strides=strides, activation=False)

        for idx, f in enumerate(filters):
            activation = True if idx < len(filters)-1 else False
            stride = 1 if idx < len(filters)-1 else strides
            x = Sep_Conv_BN(x, filters=f, strides=stride, activation=activation, deeplab=True)

    x = add([inpt, x])
    x = ReLU()(x)

    return x


# without conv_bn in id path, in-out channel keep the same(without strides)
def sep_id_block(x, filters, deeplab=False):
    inpt = x

    x = Sep_Conv_BN(x, filters, strides=1, activation=True, deeplab=deeplab)
    x = Sep_Conv_BN(x, filters, strides=1, activation=True, deeplab=deeplab)
    x = Sep_Conv_BN(x, filters, strides=1, activation=False, deeplab=deeplab)

    x = add([inpt, x])
    x = ReLU()(x)

    return x


def Sep_Conv_BN(x, filters, strides, activation=True, deeplab=False):
    if deeplab:
        # DW-BN-ReLU-PW-BN-ReLU
        x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv_BN(x, filters, kernel_size=1, strides=strides, activation=activation)
    else:
        # [DW-PW]-BN-ReLU
        x = SeparableConv2D(filters, kernel_size=3, strides=1, padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        if activation:
            x = ReLU()(x)
    return x


if __name__ == '__main__':

    # model = Xception(input_tensor=None, input_shape=(299,299,3), n_classes=1000)
    # model.summary()
    model = Xception_deeplab(input_tensor=None, input_shape=(299,299,3), n_classes=1000)
    model.summary()


