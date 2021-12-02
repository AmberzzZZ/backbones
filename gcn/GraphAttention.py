from __future__ import print_function

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
import tensorflow as tf


class GraphAttention(Layer):

    def __init__(self, out_dim,
                 attn_heads=1,
                 use_bias=True,
                 activation=None,
                 drop_rate=0.,     # transductive learning
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphAttention, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.attn_heads = attn_heads
        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.drop_rate = drop_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1], self.out_dim)    # (b,N,F)
        return output_shape

    def build(self, input_shape):
        in_dim = input_shape[-1]

        self.kernels = []
        self.biases = []
        self.att_kernels = []

        for i in range(self.attn_heads):
            # weights
            kernel = self.add_weight(shape=(in_dim, self.out_dim),        # (D,F)
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.kernels.append(kernel)

            # bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.out_dim,),              # (F,)
                                       initializer=self.bias_initializer,
                                       name='bias',
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint)
            else:
                bias = None
            self.biases.append(bias)

            # attention weights
            att_kernel = self.add_weight(shape=(self.out_dim*2, 1),        # (2F,1)
                                         initializer=self.kernel_initializer,
                                         name='att_kernel',
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint)
            self.att_kernels.append(att_kernel)

        self.built = True

    def call(self, input, mask=None):
        # input: node features, (b,N,D)

        outputs = []
        for i in range(self.attn_heads):
            kernel = self.kernels[i]
            bias = self.biases[i]
            att_kernel = self.att_kernels[i]

            # right matmul
            x = K.dot(input, kernel)    # (b,N,F)

            # attention
            att_self = att_kernel[:self.out_dim,:]     # (F,1)
            att_neighbor = att_kernel[self.out_dim:,:]   # (F,1)

            value_self = K.dot(x, att_self)             # (b,N,1)
            value_neighbor = K.dot(x, att_neighbor)     # (b,N,1)
            value = value_self + tf.transpose(value_neighbor, (0,2,1))         # (b,N,N)
            value = LeakyReLU(alpha=0.2)(value)

            # softmax
            coeff = K.softmax(value, axis=-1)   # (b,N,N)

            # weighted sum
            x = tf.matmul(coeff, x)    # (b,N,F)

            if bias:
                x += bias

            outputs.append(x)

        # concat
        x = K.concatenate(outputs, axis=-1)     # (b,N,KF)

        if self.activation:
            x = self.activation(x)

        return x

    def get_config(self):
        config = {'out_dim': self.out_dim,
                  'drop_rate': self.drop_rate,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':

    from keras.layers import Input
    from keras.models import Model
    import numpy as np

    x = Input((32,128))
    y = GraphAttention(16, attn_heads=1)(x)
    print(y)    # (b,N,out_dim)
    model = Model(x, y)

    X = np.random.uniform(0, 1, (12,32,128))

    y = model.predict(X)
    print(y.shape)    # (b,N,F)








