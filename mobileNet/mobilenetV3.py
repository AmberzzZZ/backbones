from keras.layers import Input, ReLU, Activation, Multiply, add, Conv2D, BatchNormalization, \
                         DepthwiseConv2D, GlobalAveragePooling2D, Reshape, Dense
from keras.models import Model


def MobileNetV3(input_shape=(224,224,3), input_tensor=None, n_classes=1000, se_ratio=0.25):
    if input_tensor is not None:
        inpt = input_tensor
    else:
        inpt = Input(input_shape)

    # backbone
    x = Conv_BN(inpt, 16, kernel_size=3, strides=2, activation=hard_swish)

    x = inverted_se_res_block(x, 16, kernel_size=3, strides=1, activation=relu, expansion=1, se_ratio=None)
    x = inverted_se_res_block(x, 24, kernel_size=3, strides=2, activation=relu, expansion=4, se_ratio=None)
    x = inverted_se_res_block(x, 24, kernel_size=3, strides=1, activation=relu, expansion=3, se_ratio=None)
    x = inverted_se_res_block(x, 40, kernel_size=5, strides=2, activation=relu, expansion=3, se_ratio=se_ratio)
    x = inverted_se_res_block(x, 40, kernel_size=5, strides=1, activation=relu, expansion=3, se_ratio=se_ratio)
    x = inverted_se_res_block(x, 40, kernel_size=5, strides=1, activation=hard_swish, expansion=3, se_ratio=se_ratio)
    x = inverted_se_res_block(x, 80, kernel_size=3, strides=2, activation=hard_swish, expansion=6, se_ratio=None)
    x = inverted_se_res_block(x, 80, kernel_size=3, strides=1, activation=hard_swish, expansion=2.5, se_ratio=None)
    x = inverted_se_res_block(x, 80, kernel_size=3, strides=1, activation=hard_swish, expansion=2.3, se_ratio=None)
    x = inverted_se_res_block(x, 80, kernel_size=3, strides=1, activation=hard_swish, expansion=2.3, se_ratio=None)
    x = inverted_se_res_block(x, 112, kernel_size=5, strides=1, activation=hard_swish, expansion=6, se_ratio=se_ratio)
    x = inverted_se_res_block(x, 112, kernel_size=5, strides=1, activation=hard_swish, expansion=6, se_ratio=se_ratio)
    x = inverted_se_res_block(x, 160, kernel_size=5, strides=2, activation=hard_swish, expansion=6, se_ratio=se_ratio)
    x = inverted_se_res_block(x, 160, kernel_size=5, strides=1, activation=hard_swish, expansion=6, se_ratio=se_ratio)
    x = inverted_se_res_block(x, 160, kernel_size=5, strides=1, activation=hard_swish, expansion=6, se_ratio=se_ratio)

    x = Conv_BN(x, 960, kernel_size=1, strides=1, activation=hard_swish)

    # top
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, activation='softmax')(x)

    # model
    model = Model(inpt, x)

    return model


# conv + BN + activation
def Conv_BN(x, filters, kernel_size=3, strides=1, activation=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


def DW_Conv_BN(x, kernel_size=3, strides=1, depth_multiplier=1, activation=None):
    # depth-wise
    x = DepthwiseConv2D(kernel_size, strides=strides, padding='same', depth_multiplier=depth_multiplier)(x)
    x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


def inverted_se_res_block(x, filters, kernel_size, strides, activation, expansion, se_ratio):
    inpt = x
    in_channels = x._keras_shape[-1]
    # expand: conv-bn-activation
    x = Conv_BN(x, int(in_channels*expansion), kernel_size=1, strides=1, activation=activation)
    # dw: dwconv-bn-activation
    x = DW_Conv_BN(x, kernel_size=kernel_size, strides=strides, depth_multiplier=1, activation=activation)
    # se:
    if se_ratio:
        x = se_block(x, se_ratio)
    # project: pwconv-bn
    x = Conv_BN(x, filters, kernel_size=1, strides=1, activation=None)
    # residual
    if in_channels==filters and strides==1:
        x = add([inpt, x])
    return x


def se_block(x, se_ratio):
    inpt = x
    in_channels = x._keras_shape[-1]
    # squeeze
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,in_channels))(x)
    x = Conv2D(int(in_channels*se_ratio), 1, strides=1, padding='same')(x)
    x = Activation(relu)(x)
    # excite
    x = Conv2D(in_channels, 1, strides=1, padding='same')(x)
    x = Activation(hard_sigmoid)(x)
    # reweight
    x = Multiply()([inpt, x])
    return x


def relu(x):
    return ReLU()(x)


def hard_sigmoid(x):
    return ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])


if __name__ == '__main__':

    model = MobileNetV3(input_shape=(224,224,3), input_tensor=None, n_classes=1000)
    model.summary()



