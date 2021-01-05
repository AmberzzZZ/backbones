from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D, Dropout, Activation, add
from keras.models import Model
import keras.backend as K


def squeezeNet(input_shape=(224,224,3), n_classes=1000):
    inpt = Input(input_shape)

    # stem: 7x7 conv, maxpooling
    x = Conv2D(96, 7, strides=2, padding='valid', activation='relu')(inpt)
    x = MaxPooling2D(pool_size=3, strides=2, padding='valid')(x)

    # fire2,3,4, maxpooling
    x = fire_module(x, squeeze=16, expand=64)
    x = fire_module(x, squeeze=16, expand=64)
    x = fire_module(x, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=3, strides=2, padding='valid')(x)

    # fire5,6,7,8, maxpooling
    x = fire_module(x, squeeze=32, expand=128)
    x = fire_module(x, squeeze=48, expand=192)
    x = fire_module(x, squeeze=48, expand=192)
    x = fire_module(x, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=3, strides=2, padding='valid')(x)

    # fire9
    x = fire_module(x, squeeze=64, expand=256)

    # top
    x = Dropout(0.5)(x)
    x = Conv2D(n_classes, 1, strides=1, padding='valid', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    # model
    model = Model(inpt, x)

    return model


def fire_module(x, squeeze=16, expand=64, bypass=True, complex=True):
    # 50% ratio
    skip = x
    x = Conv2D(squeeze, 1, strides=1, padding='same', activation='relu')(x)
    left = Conv2D(expand, 1, strides=1, padding='same', activation='relu')(x)
    right = Conv2D(expand, 3, strides=1, padding='same', activation='relu')(x)
    x = concatenate([left, right], axis=-1)
    if bypass:
        if K.int_shape(skip)[-1] != expand*2:
            if complex:
                skip = Conv2D(expand*2, 1, strides=1, padding='same', activation='relu')(skip)
            else:
                return x
        x = add([skip, x])
        x = Activation('relu')(x)
    return x


if __name__ == '__main__':

    model = squeezeNet()
    model.summary()






















