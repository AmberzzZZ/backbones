from keras.layers import Input, GlobalAveragePooling2D, Dense, add
from keras.models import Model
from mobilenetV1 import Conv_BN, DW_Conv_BN


def MobileNetV2(input_shape=(224,224,3), input_tensor=None, n_classes=1000):
    if input_tensor is not None:
        inpt = input_tensor
    else:
        inpt = Input(input_shape)

    # backbone
    x = Conv_BN(inpt, 32, strides=2)

    x = inverted_res_block(x, 16, strides=1, expansion=1)

    x = inverted_res_block(x, 24, strides=2, expansion=6)
    x = inverted_res_block(x, 24, strides=1, expansion=6)

    x = inverted_res_block(x, 32, strides=2, expansion=6)
    x = inverted_res_block(x, 32, strides=1, expansion=6)
    x = inverted_res_block(x, 32, strides=1, expansion=6)

    x = inverted_res_block(x, 64, strides=2, expansion=6)
    x = inverted_res_block(x, 64, strides=1, expansion=6)
    x = inverted_res_block(x, 64, strides=1, expansion=6)
    x = inverted_res_block(x, 64, strides=1, expansion=6)

    x = inverted_res_block(x, 96, strides=1, expansion=6)
    x = inverted_res_block(x, 96, strides=1, expansion=6)
    x = inverted_res_block(x, 96, strides=1, expansion=6)

    x = inverted_res_block(x, 160, strides=2, expansion=6)
    x = inverted_res_block(x, 160, strides=1, expansion=6)
    x = inverted_res_block(x, 160, strides=1, expansion=6)

    x = inverted_res_block(x, 320, strides=1, expansion=6)

    x = Conv_BN(x, 1280, 1, strides=2)

    # top
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, activation='softmax')(x)

    # model
    model = Model(inpt, x)

    return model


def inverted_res_block(x, filters, strides=1, expansion=1):
    inpt = x
    in_channels = x._keras_shape[-1]
    # expand: conv-bn-relu6
    x = Conv_BN(x, in_channels*expansion, kernel_size=1, strides=1, activation=True)
    # dw: dwconv-bn-relu6
    x = DW_Conv_BN(x, kernel_size=3, strides=strides, depth_multiplier=1)
    # project: pwconv-bn
    x = Conv_BN(x, filters, kernel_size=1, strides=1, activation=False)
    # residual
    if in_channels==filters and strides==1:
        x = add([inpt, x])
    return x


if __name__ == '__main__':

    model = MobileNetV2(input_shape=(224,224,3), input_tensor=None, n_classes=1000)
    model.summary()



