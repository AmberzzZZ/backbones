# resnet18
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, ReLU, add, GlobalAveragePooling2D, \
                         Reshape, Dense, multiply
from keras.models import Model


def resnet18(input_shape=(224,224,3), width=64, pooling=False):
    inpt = Input(input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, width, 7, strides=2, activation='relu')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    for i in range(4):
        for j in range(2):
            strides = 2 if i!=0 and j==0 else 1
            x = res_block(x, width*(2**i), strides)

    if pooling:
        x = GlobalAveragePooling2D()(x)

    # model
    model = Model(inpt, x)

    return model


def res_block(x, n_filters, strides):
    inpt = x
    # residual
    x = Conv_BN(x, n_filters, 3, strides=strides, activation='relu')
    x = Conv_BN(x, n_filters, 3, strides=1, activation=None)
    # shortcut
    if strides!=1 or inpt._keras_shape[-1]!=n_filters:
        inpt = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
    x = add([inpt, x])
    x = ReLU()(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = resnet18(input_shape=(224,224,3), width=64)
    model.summary()
    # model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

