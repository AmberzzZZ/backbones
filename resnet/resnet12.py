# resnet12
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, ReLU, add, GlobalAveragePooling2D, \
                         Reshape, Dense, multiply
from keras.models import Model


def resnet12(input_shape=(224,224,3), width=64, pooling=False):
    inpt = Input(input_shape)

    x = inpt
    # blocks
    for i in range(4):
            x = res_block(x, width*(2**i))

    if pooling:
        x = GlobalAveragePooling2D()(x)

    # model
    model = Model(inpt, x)

    return model


def res_block(x, n_filters):
    inpt = x
    # residual
    x = Conv_BN(x, n_filters, 3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 3, strides=1, activation=None)
    # shortcut
    if inpt._keras_shape[-1]!=n_filters:
        inpt = Conv_BN(inpt, n_filters, 1, strides=1, activation=None)
    x = add([inpt, x])
    x = ReLU()(x)
    # maxpooling
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = resnet12(input_shape=(224,224,3), width=64)
    model.summary()
    # model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

