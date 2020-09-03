from keras.layers import Input, Conv3D, BatchNormalization, ReLU, add, MaxPooling3D
from keras.models import Model


n_blocks = {50: [3,4,6,3], 101: [3,4,23,3]}
n_filters = [256, 512, 1024, 2048]


def p3d_resnet(input_shape=(5,256,256,1), depth=50):
    inpt = Input(input_shape)

    # stem: conv+bn+relu+pool
    x = ConvBN(inpt, 64, 3, strides=2, activation='relu')
    x = ConvBN(x, 64, 3, strides=1, activation='relu')
    x = MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[depth]
    block_variants = [p3d_resblock_a, p3d_resblock_b, p3d_resblock_c]
    cnt = 0
    for i in range(len(num_blocks)):
        for j in range(num_blocks[i]):
            strides = 2 if i!=0 and j==0 else 1
            x = block_variants[cnt%3](x, n_filters[i], strides)

    # model
    model = Model(inpt, x)

    return model


def p3d_resblock_a(inpt, filters, strides=1, padding='same'):
    x = ConvBN(inpt, filters//4, kernel_size=1, strides=strides, activation='relu')
    x = ConvBN(x, filters//4, kernel_size=(1,3,3), activation='relu')
    x = ConvBN(x, filters//4, kernel_size=(3,1,1), activation='relu')
    x = ConvBN(x, filters, kernel_size=1, activation=None)

    if strides!=1 or inpt._keras_shape[-1]!=filters:
        inpt = ConvBN(inpt, filters, 1, strides=strides, activation=None)
    x = add([x, inpt])
    x = ReLU()(x)
    return x


def p3d_resblock_b(inpt, filters, strides=1, padding='same'):
    x = ConvBN(inpt, filters//4, kernel_size=1, strides=strides, activation='relu')
    x1 = ConvBN(x, filters//4, kernel_size=(1,3,3), activation='relu')
    x2 = ConvBN(x, filters//4, kernel_size=(3,1,1), activation='relu')
    x = add([x1, x2])
    x = ConvBN(x, filters, kernel_size=1, activation=None)

    if strides!=1 or inpt._keras_shape[-1]!=filters:
        inpt = ConvBN(inpt, filters, 1, strides=strides, activation=None)
    x = add([x, inpt])
    x = ReLU()(x)
    return x


def p3d_resblock_c(inpt, filters, strides=1, padding='same'):
    x = ConvBN(inpt, filters//4, kernel_size=1, strides=strides, activation='relu')
    x1 = ConvBN(x, filters//4, kernel_size=(1,3,3), activation='relu')
    x2 = ConvBN(x1, filters//4, kernel_size=(3,1,1), activation=None)
    x = add([x1, x2])
    x = ReLU()(x)
    x = ConvBN(x, filters, kernel_size=1, activation=None)

    if strides!=1 or inpt._keras_shape[-1]!=filters:
        inpt = ConvBN(inpt, filters, 1, strides=strides, activation=None)
    x = add([x, inpt])
    x = ReLU()(x)
    return x


def ConvBN(x, filters, kernel_size, strides=1, padding='same', activation=None):
    x = Conv3D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = p3d_resnet(input_shape=(5,256,256,1), depth=50)
    model.summary()

