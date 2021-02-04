# resnext, 50 & 101, use self defined groupConv Layer
from keras.layers import Input, MaxPooling2D, ReLU, add, BatchNormalization, GlobalAveragePooling2D, \
                         Dense, Reshape, multiply, Lambda, Softmax
from keras.models import Model
import keras.backend as K
from groupConv import GroupConv2D


n_blocks = {50: [3,4,6,3], 101: [3,4,23,3]}
g_filters = [128, 256, 512, 1024]          # group filters for resnext with C=32
n_filters = [256, 512, 1024, 2048]         # output filters


def SKNet(input_shape=(224,224,3), depth=50, r=16, C=32, M=2):
    inpt = Input(shape=input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 7, strides=2, activation='relu')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[depth]
    for i in range(len(num_blocks)):
        for j in range(num_blocks[i]):
            strides = 2 if i!=0 and j==0 else 1
            x = sk_unit(x, g_filters[i], n_filters[i], strides, M, C, r)
    # model
    model = Model(inpt, x)

    return model


def sk_unit(x, g_filters, n_filters, strides, M=2, C=32, se_ratio=16):
    inpt = x
    # 1x1conv-3x3sk_conv-1x1conv
    x = Conv_BN(inpt, g_filters, 1, strides=strides, activation='relu')
    x = Conv_SK(x, g_filters, M, C, se_ratio=se_ratio)
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    # shortcut
    if strides!=1 or inpt._keras_shape[-1]!=n_filters:
        inpt = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
    x = add([inpt, x])
    x = ReLU()(x)
    return x


def Conv_SK(inpt, n_filters, M=2, C=32, se_ratio=16, dilated=False):
    if dilated:
        kernel_sizes = [3 for i in range(M)]
        dilation_rates = [i*2 for i in range(M)]
    else:
        kernel_sizes = [3+i*2 for i in range(M)]
        dilation_rates = [1 for i in range(M)]
    # split: group/depth conv-BN-relu
    features = []
    for i in range(M):
        x = Conv_BN(inpt, n_filters, kernel_sizes[i], strides=1, dilation_rate=dilation_rates[i],
                    group=C, activation='relu')
        features.append(x)
    stacked_features = Lambda(lambda x: K.stack(x, axis=-1))(features)
    # fuse: add-GAP-fc-BN-relu
    x = add(features)
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_filters//se_ratio, activation=None, use_bias=False)(x)
    x = BatchNormalization()(x)
    fused_feature = ReLU()(x)
    # select: fc-sigmoid, reweight sum
    vecs = []
    for i in range(M):
        x = Dense(n_filters, activation=None, use_bias=True)(fused_feature)
        vecs.append(x)
    stacked_vecs = Lambda(lambda x: K.stack(x, axis=-1))(vecs)
    stacked_vecs = Softmax(axis=-1)(stacked_vecs)
    stacked_vecs = Reshape((1,1,n_filters,M))(stacked_vecs)
    x = multiply([stacked_features, stacked_vecs])
    x = Lambda(lambda x: K.sum(x, axis=-1))(x)

    return x


def Conv_BN(x, n_filters, kernel_size, strides, dilation_rate=1, group=1, activation=None):
    x = GroupConv2D(n_filters, kernel_size, strides=strides, padding='same', group=group,
                    dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = SKNet(input_shape=(224,224,3), depth=50)
    model.summary()
    model.save_weights('sknet.h5')

    # x = Input((56,56,128))
    # y = Conv_SK(x, n_filters=128, dilated=False)
    # print(y)



