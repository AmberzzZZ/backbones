from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, concatenate,\
                         AveragePooling2D, GlobalAveragePooling2D, Dense, Lambda
from keras.models import Model


n_blocks = {'dense121': [6,12,24,16],
            'dense169': [6,12,32,32],
            'dense201': [6,12,48,32],
            'dense264': [6,12,64,48]}


def csp_densenet(input_tensor=None, input_shape=(224,224,3), n_classes=1000,
                 back='dense121', bottleneck=False, K=32, theta=1.0, model='b'):
    if model=='b':   # csp-densent
        transition_dense = True
        transition_fuse = True
    if model=='c':    # fusion first
        transition_dense = False
        transition_fuse = True
    if model=='d':    # fusion last
        transition_dense = True
        transition_fuse = False

    if input_tensor is not None:
        inpt = input_tensor
    else:
        inpt = Input(input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 7, strides=2, activation='relu')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[back]
    for i in range(len(num_blocks)):
        # partial
        in_filters = x._keras_shape[-1]
        skip = Lambda(lambda x: x[..., :in_filters//2])(x)
        x = Lambda(lambda x: x[..., in_filters//2:])(x)
        # dense block
        for j in range(num_blocks[i]):
            # dense block
            if bottleneck:
                x = dense_block_b(x, K)      # [N,h,w,nK]
            else:
                x = dense_block(x, K)      # [N,h,w,nK]
        # transition
        if transition_dense and i != len(num_blocks)-1:    # last block but not last layer
            # transition block
            x = csp_transition_block(x, theta)
        # fusion
        x = concatenate([skip, x], axis=-1)
        if transition_fuse and i != len(num_blocks)-1:
            x = transition_block(x, theta)

    # for last dense block
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)

    # head
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, activation='softmax')(x)

    # model
    model = Model(inpt, x)
    return model


# 3x3 conv, BN-relu-conv-concat
def dense_block(x, K):
    inpt = x
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = Conv2D(K, 3, strides=1, padding='same')(x)
    x = concatenate([inpt, x])
    return x


# 1x1 conv + 3x3 conv, BN-relu-conv-concat
def dense_block_b(x, K):
    inpt = x
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = Conv2D(4*K, 1, strides=1, padding='same')(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = Conv2D(K, 3, strides=1, padding='same')(x)
    x = concatenate([inpt, x])
    return x


# 1x1 conv + 2x2 avg pooling
def transition_block(x, theta):
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = Conv2D(int(theta*x._keras_shape[-1]), 1, strides=1, padding='same')(x)
    x = AveragePooling2D(pool_size=2, strides=2, padding='same')(x)
    return x


# 1x1 conv
def csp_transition_block(x, theta):
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = Conv2D(int(theta*x._keras_shape[-1]), 1, strides=1, padding='same')(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = csp_densenet(input_shape=(224,224,3))
    model.summary()


