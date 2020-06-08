from keras.layers import Input, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D,  \
                         GlobalAveragePooling2D, Reshape, Dropout, Activation
from keras.models import Model


def MobileNetV1(input_shape=(224,224,3), input_tensor=None, n_classes=1000):
    if input_tensor is not None:
        inpt = input_tensor
    else:
        inpt = Input(input_shape)

    # backbone
    x = Conv_BN(inpt, 32, strides=2)

    x = DW_PW_block(x, 64, strides=1)

    x = DW_PW_block(x, 128, strides=2)
    x = DW_PW_block(x, 128, strides=1)

    x = DW_PW_block(x, 256, strides=2)
    x = DW_PW_block(x, 256, strides=1)

    x = DW_PW_block(x, 512, strides=2)
    x = DW_PW_block(x, 512, strides=1)
    x = DW_PW_block(x, 512, strides=1)
    x = DW_PW_block(x, 512, strides=1)
    x = DW_PW_block(x, 512, strides=1)
    x = DW_PW_block(x, 512, strides=1)

    x = DW_PW_block(x, 1024, strides=2)
    x = DW_PW_block(x, 1024, strides=1)

    # top
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,x._keras_shape[-1]))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(n_classes, 1, strides=1, padding='same')(x)
    x = Reshape((x._keras_shape[-1],))(x)
    x = Activation('softmax')(x)

    # model
    model = Model(inpt, x)

    return model


# conv + BN + relu6
def Conv_BN(x, filters, kernel_size=3, strides=1, activation=True):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU(6.)(x)
    return x


def DW_Conv_BN(x, kernel_size=3, strides=1, depth_multiplier=1):
    # depth-wise
    x = DepthwiseConv2D(kernel_size, strides=strides, padding='same', depth_multiplier=depth_multiplier)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    return x


def DW_PW_block(x, filters, strides=1):
    # DW
    x = DW_Conv_BN(x, strides=strides)
    # PW
    x = Conv_BN(x, filters, kernel_size=1, strides=1, activation=True)
    return x


if __name__ == '__main__':

    model = MobileNetV1(input_shape=(224,224,3), input_tensor=None, n_classes=1000)
    model.summary()



