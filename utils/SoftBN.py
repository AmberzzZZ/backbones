from keras.engine import Layer
from keras import initializers
from keras import regularizers
from keras import constraints
import keras.backend as K
import tensorflow as tf


class SoftBatchNormalization(Layer):
    # [new arg] steps_per_update: how many batch to update gradient
    # [new arg] iterations: counter
    def __init__(self,
                 steps_per_update=1,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(SoftBatchNormalization, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        self.steps_per_update = steps_per_update
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = scale
        self.center = center
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)

    def build(self, input_shape):
        shape = (input_shape[self.axis], )
        # untrainable moving mean&variance
        self.moving_mean = self.add_weight(name='moving_mean',
                                           shape=shape,
                                           initializer=self.moving_mean_initializer,
                                           trainable=False)

        self.moving_variance = self.add_weight(name='moving_variance',
                                               shape=shape,
                                               initializer=self.moving_variance_initializer,
                                               trainable=False)
        self.steps_mean = self.add_weight(name='steps_mean',
                                          shape=shape,
                                          initializer=self.moving_mean_initializer,
                                          trainable=False)
        self.steps_variance = self.add_weight(name='steps_variance',
                                              shape=shape,
                                              initializer=self.moving_variance_initializer,
                                              trainable=False)
        # trainable rescale gamma&beta
        if self.scale:
            self.gamma = self.add_weight(name='gamma',
                                         shape=shape,
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint,
                                         trainable=True)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(name='beta',
                                        shape=shape,
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        trainable=True)
        else:
            self.beta = None

    def call(self, inputs, training=None):
        if not self.trainable:
            training = False
        else:
            # The learning phase flag is a bool tensor (0 = test, 1 = train)
            training = K.learning_phase()

        if training is not False:
            K.update_add(self.iterations, 1)
            # compute current mean&var
            mini_mean, mini_variance = tf.nn.moments(inputs, axes=[0,1,2])
            # affine the inputs
            x = (inputs - self.steps_mean) / K.sqrt(self.steps_variance + self.epsilon)
            x = self.gamma * x + self.beta
            # update the moving params
            K.moving_average_update(self.moving_mean, mini_mean, self.momentum)
            K.moving_average_update(self.moving_variance, mini_variance, self.momentum)
            # update the short-term params under specific condition
            cond = K.equal(self.iterations % self.steps_per_update, 0)
            K.switch(cond, lambda: self.steps_mean*0, K.update_add(self.steps_mean, mini_mean))
            K.switch(cond, lambda: self.steps_variance*0, K.update_add(self.steps_variance, mini_mean))
        else:
            # affine
            scale = self.gamma / K.sqrt(self.moving_variance + self.epsilon)
            x = inputs * scale + (self.beta - self.moving_mean * scale)
        return x

    def get_config(self):
        config = {
            'steps_per_update': self.steps_per_update,
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'moving_mean_initializer': constraints.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': constraints.serialize(self.moving_variance_initializer)
        }
        base_config = super(SoftBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':

    from keras.layers import Input, Dense, GlobalAveragePooling2D
    from keras.models import Model
    from keras.optimizers import Adam
    import numpy as np

    x = Input((32,32,1))
    y = SoftBatchNormalization(steps_per_update=4, axis=-1, momentum=0.99, epsilon=1e-3)(x)
    y = GlobalAveragePooling2D()(y)
    y = Dense(3, activation='softmax')(y)
    model = Model(x, y)
    model.compile(Adam(1e-4), 'mse')

    X = np.random.uniform(0, 1., (1000,32,32,1))
    Y = np.random.randint(0, 1, (1000,3), dtype='int32')
    model.fit(X, Y, epochs=10, batch_size=100)








