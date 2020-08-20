from keras.layers import Input, Conv2D, ReLU, MaxPool2D


def vgg16_back(inpt):

    # conv1
    x = conv_block(inpt, 64, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv2
    x = conv_block(x, 128, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv3
    x = conv_block(x, 256, 3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv4
    conv4 = conv_block(x, 512, 3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(conv4)

    # conv5: 3x3 s2 pooling
    x = conv_block(x, 512, 3)
    x = MaxPool2D(pool_size=3, strides=1, padding='same')(x)

    # conv6: atrous conv
    x = Conv2D(1024, 3, strides=1, padding='same', dilation_rate=(6, 6))(x)

    # conv7: 1x1 conv
    conv7 = Conv2D(1024, 1, strides=1, padding='same')(x)

    # conv8: 1x1x256 conv + 3x3x512 s2 conv
    x = Conv2D(256, 1, strides=1, padding='same')(conv7)
    conv8 = Conv2D(512, 3, strides=2, padding='same')(x)

    # conv9: 1x1x128 conv + 3x3x256 s2 conv
    x = Conv2D(128, 1, strides=1, padding='same')(conv8)
    conv9 = Conv2D(256, 3, strides=2, padding='same')(x)

    # conv10: 1x1x128 conv + 3x3x256 s1 p0 conv
    x = Conv2D(128, 1, strides=1, padding='same')(conv9)
    conv10 = Conv2D(256, 3, strides=1, padding='valid')(x)

    # conv11: 1x1x128 conv + 3x3x256 s1 p0 conv
    x = Conv2D(128, 1, strides=1, padding='same')(conv10)
    conv11 = Conv2D(256, 3, strides=1, padding='valid')(x)

    return [conv4, conv7, conv8, conv9, conv10, conv11]


def conv_block(x, filters, n_layers, kernel_size=3, strides=1):
    for i in range(n_layers):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    inpt = Input((300,300,3))
    features = vgg16_back(inpt)
    print(features)








