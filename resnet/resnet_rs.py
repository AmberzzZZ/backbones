# resnet 50 & 101
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, add, GlobalAveragePooling2D, \
                         Reshape, Dense, multiply, Dropout, AveragePooling2D, Layer
from keras.models import Model
from keras.regularizers import l2


n_blocks = {50: [3,4,6,3], 101: [3,4,23,3], 152: [3,8,36,3], 200: [3,24,36,3],
            270: [4,29,53,4], 350: [4,36,72,4], 420: [4,44,87,4]}
n_filters = [256, 512, 1024, 2048]
KERNEL_REGULIZER = l2(4e-5)


def resnet_rs(input_shape=(224,224,3), n_classes=1000, depth=50, se_ratio=.25,
              drop_rate=0.25, stochastic_rate=0.0, bn_momentum=0.9999):
    inpt = Input(input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 3, strides=2, activation='relu', bn_momentum=bn_momentum)
    x = Conv_BN(x, 64, 3, strides=1, activation='relu', bn_momentum=bn_momentum)
    x = Conv_BN(x, 64, 3, strides=1, activation='relu', bn_momentum=bn_momentum)

    # blocks
    num_blocks = n_blocks[depth]
    b = 0
    total_blocks = sum(num_blocks)
    for i in range(len(num_blocks)):
        for j in range(num_blocks[i]):
            strides = 2 if j==0 else 1
            stochastic_drop = stochastic_rate * b / total_blocks
            x = res_block(x, n_filters[i], strides, se_ratio, stochastic_drop, bn_momentum)
            b += 1

    x = GlobalAveragePooling2D()(x)
    if drop_rate:
        x = Dropout(drop_rate)(x)
    x = Dense(n_classes, activation='softmax', kernel_regularizer=KERNEL_REGULIZER,
              bias_regularizer=KERNEL_REGULIZER)(x)

    # model
    model = Model(inpt, x)

    return model


def res_block(x, n_filters, strides, se_ratio=0.25, stochastic_rate=0., bn_momentum=0.9999):
    inpt = x
    # residual
    x = Conv_BN(x, n_filters//4, 1, strides=1, activation='relu', bn_momentum=bn_momentum)
    x = Conv_BN(x, n_filters//4, 3, strides=strides, activation='relu', bn_momentum=bn_momentum)
    x = Conv_BN(x, n_filters, 1, strides=1, bn=False, activation=None)
    # se
    if se_ratio:
        x = SE_block(x, se_ratio)
    # stochastic depth
    if stochastic_rate:
        x = Dropout(stochastic_rate, noise_shape=(1,1,1,1))(x)   # (None,1,1,1)

    # shortcut
    if strides!=1:
        inpt = AveragePooling2D(pool_size=2, strides=2, padding='same')(inpt)
        inpt = Conv_BN(inpt, n_filters, 1, strides=1, bn=False, activation=None)

    x = add([inpt, x])
    x = BatchNormalization(momentum=bn_momentum)(x)
    x = ReLU()(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, bn=True, activation=None, bn_momentum=0.9999):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False,
               kernel_regularizer=KERNEL_REGULIZER)(x)
    if bn:
        x = BatchNormalization(momentum=bn_momentum)(x)
    if activation:
        x = ReLU()(x)
    return x


class WeightDecayL2(Layer):
    def __init__(self, weight_decay=0.):
        super(WeightDecayL2, self).__init__()
        self.weight_decay = weight_decay

    def call(self, input):
        




def SE_block(inpt, ratio=.25):     # spatial squeeze and channel excitation
    x = inpt
    n_filters = x._keras_shape[-1]
    x = GlobalAveragePooling2D()(x)
    x = Dense(int(n_filters*ratio), activation='relu', use_bias=True,
              kernel_regularizer=KERNEL_REGULIZER,
              bias_regularizer=KERNEL_REGULIZER)(x)
    x = Dense(n_filters, activation='sigmoid', use_bias=True,
              kernel_regularizer=KERNEL_REGULIZER,
              bias_regularizer=KERNEL_REGULIZER)(x)
    x = Reshape((1,1,n_filters))(x)
    x = multiply([inpt, x])
    return x


def resnet_rs50(n_classes):
    return resnet_rs(input_shape=(160,160,3), n_classes=n_classes, depth=50,
                     drop_rate=0.25, stochastic_rate=0.)


def resnet_rs101(n_classes):
    # input_shape = [160, 192]
    return resnet_rs(input_shape=(160,160,3), n_classes=n_classes, depth=101,
                     drop_rate=0.25, stochastic_rate=0.)


def resnet_rs152(n_classes):
    # input_shape = [192, 224, 256]
    return resnet_rs(input_shape=(192,192,3), n_classes=n_classes, depth=152,
                     drop_rate=0.25, stochastic_rate=0.)


def resnet_rs200(n_classes):
    return resnet_rs(input_shape=(256,256,3), n_classes=n_classes, depth=200,
                     drop_rate=0.25, stochastic_rate=0.2)


def resnet_rs270(n_classes):
    return resnet_rs(input_shape=(256,256,3), n_classes=n_classes, depth=270,
                     drop_rate=0.25, stochastic_rate=0.2)


def resnet_rs350(n_classes):
    # input_shape = [256, 320]
    return resnet_rs(input_shape=(256,256,3), n_classes=n_classes, depth=350,
                     drop_rate=0.4, stochastic_rate=0.2)


def resnet_rs420(n_classes):
    return resnet_rs(input_shape=(320,320,3), n_classes=n_classes, depth=420,
                     drop_rate=0.4, stochastic_rate=0.2)


if __name__ == '__main__':

    model = resnet_rs(n_classes=1000, stochastic_rate=.2)
    model.summary()
    # model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

