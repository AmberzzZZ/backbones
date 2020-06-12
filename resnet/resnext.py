# resnext, 50 & 101, use self defined groupConv Layer
from keras.layers import Input, MaxPooling2D, ReLU, add, BatchNormalization
from keras.models import Model
from groupConv import GroupConv2D


n_blocks = {50: [3,4,6,3], 101: [3,4,23,3]}
g_filters = [128, 256, 512, 1024]          # group filters for resnext with C=32
n_filters = [256, 512, 1024, 2048]         # output filters


def resnext(input_shape=(224,224,3), depth=50, C=32):
    inpt = Input(shape=input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 7, strides=2, activation='relu')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[depth]
    for i in range(len(num_blocks)):
        for j in range(num_blocks[i]):
            strides = 2 if i!=0 and j==0 else 1
            x = resnext_block(x, g_filters[i], n_filters[i], strides, C)

    # model
    model = Model(inpt, x)

    return model


def resnext_block(x, g_filters, n_filters, strides, C=32):
    inpt = x
    # 1x1conv-3x3group conv-1x1conv
    x = Conv_BN(inpt, g_filters, 1, strides=strides, activation='relu')
    x = Conv_BN(x, g_filters, 3, strides=1, group=C, activation='relu')
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    # shortcut
    if strides!=1 or inpt._keras_shape[-1]!=n_filters:
        inpt = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
    x = add([inpt, x])
    x = ReLU()(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, group=1, activation=None):
    x = GroupConv2D(n_filters, kernel_size, strides=strides, padding='same', group=group)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x



if __name__ == '__main__':

    model = resnext(input_shape=(224,224,3), depth=50)
    model.summary()



