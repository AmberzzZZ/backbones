from keras.layers import Input, Conv2D, ReLU, Flatten, Dense, MaxPool2D
from keras.models import Model


def vgg16_model(input_shape=(224,224,3), n_classes=10):

    inpt = Input(input_shape)

    x = conv_block(inpt, 64, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    x = conv_block(x, 128, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    x = conv_block(x, 256, 3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    x = conv_block(x, 512, 3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    x = conv_block(x, 512, 3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    print(x)

    x = Flatten()(x)
    x = Dense(4096)(x)
    x = Dense(4096)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inpt, x)

    return model


def vgg19_model(input_shape=(224,224,3), n_classes=10):

    inpt = Input(input_shape)

    x = conv_block(inpt, 64, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    x = conv_block(x, 128, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    x = conv_block(x, 256, 4)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    x = conv_block(x, 512, 4)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    x = conv_block(x, 512, 4)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    x = Flatten()(x)
    x = Dense(4096)(x)
    x = Dense(4096)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inpt, x)

    return model


def conv_block(x, filters, n_layers, kernel_size=3, strides=1):
    for i in range(n_layers):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = vgg16_model(input_shape=(224,224,3), n_classes=10)
    # model.summary()


