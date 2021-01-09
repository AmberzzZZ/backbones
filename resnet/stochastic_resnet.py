# resnet 50 & 101
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, ReLU, add, GlobalAveragePooling2D, \
                         Lambda
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras.initializers import Constant
import numpy as np


n_blocks = {50: [3,4,6,3], 101: [3,4,23,3]}
n_filters = [256, 512, 1024, 2048]


def stochastic_resnet(input_shape=(224,224,3), depth=50, pooling=False, training=True):
    inpt = Input(input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 7, strides=2, activation='leaky')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[depth]
    probs = np.linspace(1, 0.5, num=sum(num_blocks))    # linear decay probs for testing

    for i in range(len(num_blocks)):
        for j in range(num_blocks[i]):
            strides = 2 if i!=0 and j==0 else 1
            x = res_block(x, n_filters[i], strides, probs[num_blocks[i]+j], training)

    if pooling:
        x = GlobalAveragePooling2D()(x)

    # model
    model = Model(inpt, x)

    return model


def res_block(inpt, n_filters, strides, prob, training):
    if training:
        # residual
        x = inpt
        x = Conv_BN(x, n_filters//4, 1, strides=strides, activation='relu')
        x = Conv_BN(x, n_filters//4, 3, strides=1, activation='relu')
        residual = Conv_BN(x, n_filters, 1, strides=1, activation=None)
        # shortcut
        if strides!=1 or K.int_shape(inpt)[-1]!=n_filters:
            skip = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
        else:
            skip = inpt
        active = RandomBinomial(prob)(skip)    # [0,1] switch for training
        x = Lambda(lambda x: tf.cond(active>0, lambda: ReLU()(add([x[0],x[1]])), lambda: x[1]))([residual,skip])
        return x

    else:
        # residual
        x = inpt
        x = Conv_BN(x, n_filters//4, 1, strides=strides, activation='relu')
        x = Conv_BN(x, n_filters//4, 3, strides=1, activation='relu')
        x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
        residual = Lambda(lambda x: prob*x)(x)
        # shortcut
        if strides!=1 or inpt._keras_shape[-1]!=n_filters:
            skip = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
        else:
            skip = inpt
        x = add([residual, skip])
        x = ReLU()(x)
        return x


class RandomBinomial(Layer):

    def __init__(self, prob, **kwargs):
            self.prob = prob
            super(RandomBinomial, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(name='bernoulli_factor',
                                      shape=(1,),
                                      initializer=Constant(1.),
                                      trainable=False)
        super(RandomBinomial, self).build(input_shape)

    def call(self, x):
        active = tf.constant(np.random.binomial(1, self.prob))
        return active

    def compute_output_shape(self, input_shape):
        return (1,)


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    # posibilities = np.linspace(1, 0.5, num=sum(n_blocks[50]))
    # bernoulli_swithes = np.random.binomial(1, posibilities)
    # print(posibilities)
    # print(bernoulli_swithes)

    model = stochastic_resnet(input_shape=(224,224,3), depth=50, training=True)
    model.summary()

