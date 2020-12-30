from keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D, MaxPooling2D, add
import keras.backend as K
from keras.models import Model
from keras.losses import mean_squared_error
from keras.optimizers import RMSprop


def hourglass(input_shape=(256,256,3), n_classes=17, n_stacks=2, n_channles=256):
    inpt = Input(input_shape)

    # stem: 7x7 conv, residual, maxpooling, residual, residual
    x = Conv_BN(inpt, 64, 3, strides=2, activation='relu')
    x = residual(x, n_channles//2)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = residual(x, n_channles//2)
    x = residual(x, n_channles)

    # stack hourglass modules
    outputs = []
    for i in range(n_stacks):
        x, x_intermediate = hourglass_module(x, n_classes, n_channles)
    outputs.append(x_intermediate)

    # model
    model = Model(inpt, outputs)
    model.compile(RMSprop(lr=5e-4), loss=mean_squared_error, metrics=['acc'])

    return model


def hourglass_module(inpt, n_classes, n_channles):
    features = []
    x = inpt
    # encoder: residual + maxpooling
    x = residual(x, n_channles)         # x4
    features.append(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = residual(x, n_channles)        # x8
    features.append(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = residual(x, n_channles)        # x16
    features.append(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = residual(x, n_channles)        # x32
    features.append(x)

    # mid connection: 3 residuals + skip
    skip = x
    skip = residual(x, n_channles)
    x = residual(x, n_channles)
    x = residual(x, n_channles)
    x = residual(x, n_channles)
    x = add([x, skip])

    # decoder: skip: residual, feature: upSamp, add, residual
    skip = residual(features[2], n_channles)
    x = UpSampling2D()(x)
    x = add([x, skip])
    x = residual(x, n_channles)

    skip = residual(features[1], n_channles)
    x = UpSampling2D()(x)
    x = add([x, skip])
    x = residual(x, n_channles)

    skip = residual(features[0], n_channles)
    x = UpSampling2D()(x)
    x = add([x, skip])
    x = residual(x, n_channles)

    # head branches
    x = Conv_BN(x, n_channles, 1, strides=1, activation='relu')
    output_branch = Conv2D(n_channles, 1, strides=1, padding='same')(x)
    x_intermediate = Conv2D(n_classes, 1, strides=1, padding='same')(x)
    intermediate_branch = Conv2D(n_channles, 1, strides=1, padding='same')(x_intermediate)
    x = add([inpt, output_branch, intermediate_branch])

    return x, x_intermediate


def residual(inpt, n_filters, strides=1):
    # bottleneck: 1x1, 3x3, 1x1
    x = inpt
    x = Conv_BN(x, n_filters//2, 1, strides=strides, activation='relu')
    x = Conv_BN(x, n_filters//2, 3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    # skip: 1x1 conv
    if K.int_shape(inpt)[-1] == n_filters and strides==1:
        # identity
        skip = inpt
    else:
        # 1x1 conv
        skip = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
    # add
    x = add([x, skip])
    x = ReLU()(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = hourglass((256,256,3))
    model.summary()
    model.save("hourglass.h5")









