from __future__ import print_function

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
import tensorflow as tf


class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, out_dim,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]     # (b,N,D)
        output_shape = (features_shape[0], features_shape[1], self.out_dim)    # (b,N,F)
        return output_shape

    def build(self, input_shapes):
        in_dim = input_shapes[0][-1]

        self.kernel = self.add_weight(shape=(in_dim, self.out_dim),        # (D,F)
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.out_dim,),              # (F,)
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0]   # (b,N,D)
        adj = inputs[1]        # (b,N,N)
        # left matmul: batch dot
        x = tf.matmul(adj, features)   # (b,N,D)
        # right matmul: batch dot
        x = K.dot(x, self.kernel)    # (b,N,F)
        # bias & act
        if self.bias:
            x += self.bias
        return self.activation(x)

    def get_config(self):
        config = {'out_dim': self.out_dim,
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

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ResGraphConvolution(Layer):
    """graph convolution layer with a residual path"""
    def __init__(self, out_dim,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ResGraphConvolution, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]     # (b,N,D)
        output_shape = (features_shape[0], features_shape[1], self.out_dim)    # (b,N,F)
        return output_shape

    def build(self, input_shapes):
        in_dim = input_shapes[0][-1]

        self.kernel_id = self.add_weight(shape=(in_dim, self.out_dim),        # (D,F)
                                      initializer=self.kernel_initializer,
                                      name='kernel_id',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernel = self.add_weight(shape=(in_dim, self.out_dim),        # (D,F)
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias_id = self.add_weight(shape=(self.out_dim,),              # (F,)
                                        initializer=self.bias_initializer,
                                        name='bias_id',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.bias = self.add_weight(shape=(self.out_dim,),              # (F,)
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0]   # (b,N,D)
        adj = inputs[1]        # (b,N,N)

        # id path
        id = K.dot(features, self.kernel_id)
        if self.bias:
            id += self.bias_id
        # residual path
        # left matmul: batch dot
        x = tf.matmul(adj, features)   # (b,N,D)
        # right matmul: batch dot
        x = K.dot(x, self.kernel)    # (b,N,F)
        # bias & act
        if self.bias:
            x += self.bias
        return self.activation(id+x)

    def get_config(self):
        config = {'out_dim': self.out_dim,
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

        base_config = super(ResGraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':

    from keras.layers import Input
    from keras.models import Model
    import numpy as np

    x = Input((32,128))
    adj = Input((32,32))
    # y = GraphConvolution(16)([x,adj])
    # print(y)
    y = ResGraphConvolution(16)([x,adj])
    print(y)
    model = Model([x,adj], y)

    X = np.random.uniform(0, 1, (12,32,128))
    adj = np.random.uniform(0, 1, (12,32,32))

    y = model.predict([X,adj])
    print(y.shape)    # (b,N,F)








