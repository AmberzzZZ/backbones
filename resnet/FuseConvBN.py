from keras.engine import Layer
import keras.backend as K
import tensorflow as tf
import numpy as np


class FuseConvBN(Layer):

    # F = W_{bn} * (W_{conv} * F) + b_{bn}
    # W_{bn} is a diag mat computed from BN params
    # b_{bn} is a vec mat computed from BN params

    def __init__(self, beta, gamma, mean, variance, conv_weights, strides=(1,1), padding='same', epsilon=1e-3,
                 dilation_rate=(1,1), **kwargs):
        super(FuseConvBN, self).__init__(**kwargs)
        # conv layer config
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        # origin weights
        self.beta = tf.constant(beta, dtype='float32')
        self.gamma = tf.constant(gamma, dtype='float32')
        self.mean = tf.constant(mean, dtype='float32')
        self.variance = tf.constant(variance, dtype='float32')
        self.conv_weights = tf.constant(conv_weights, dtype='float32')

        # compute W_{bn} & b_{bn}
        k, k, filters_in, filters_out = K.int_shape(self.conv_weights)  # build shape in keras Conv
        self.filters_out = filters_out
        weights_conv = tf.reshape(tf.transpose(self.conv_weights, (3,0,1,2)), (filters_out,-1))
        weights_bn = tf.matrix_diag(gamma/tf.sqrt(variance+epsilon))
        bias_bn = beta - gamma*mean/tf.sqrt(variance+epsilon)

        # compute fused W & b
        fused_weights = tf.matmul(weights_bn,weights_conv)
        self.fused_weights = tf.transpose(tf.reshape(fused_weights, (filters_out,k,k,filters_in)), (1,2,3,0))
        self.fused_bias = bias_bn

    def call(self, x):
        # run conv using fused W & b
        x = K.conv2d(x, self.fused_weights, strides=self.strides, padding=self.padding, dilation_rate=self.dilation_rate)
        x = K.bias_add(x, self.fused_bias)
        return x

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        output_h = int(h / self.strides[0])
        output_w = int(h / self.strides[1])
        return b,output_h,output_w,self.filters_out


if __name__ == '__main__':

    from keras.layers import Input, Conv2D, BatchNormalization
    from keras.models import Model

    x = Input((32,32,4))
    conv_weights = np.ones((3,3,4,16), dtype=np.float32)
    beta = np.ones((16,), dtype=np.float32)
    gamma = np.ones((16,), dtype=np.float32)
    mean = np.ones((16,), dtype=np.float32)
    variance = np.ones((16,), dtype=np.float32)
    y = FuseConvBN(beta, gamma, mean, variance, conv_weights, strides=(1,1), padding='same', dilation_rate=(1,1))(x)
    fuse_model = Model(x, y)

    y = Conv2D(16, 3, strides=1, padding='same', activation=None, use_bias=False, weights=[conv_weights])(x)
    y = BatchNormalization(weights=[gamma, beta, mean, variance])(y)
    normal_model = Model(x, y)

    inpt = np.random.uniform(0, 1, (1,32,32,4))
    y1 = fuse_model.predict(inpt)[0]
    y2 = normal_model.predict(inpt)[0]

    print(y1.shape, y2.shape)

    print(np.array_equal(y1, y2))
    print(np.sum(y1-y2), np.max(y1-y2), np.min(y1-y2))
    print(y1[0,0])
    print(y2[0,0])












