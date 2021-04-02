from keras.optimizers import Optimizer
import keras.backend as K
import tensorflow as tf


class SGDW(Optimizer):
    # SGD with weight_decay & EMA
    def __init__(self,
                 lr=0.01,
                 momentum=0.,
                 decay=0.,
                 weight_decay=4e-5,
                 ema_momentum=0.9999,
                 nesterov=False,
                 **kwargs):
        super(SGDW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.ema_momentum = ema_momentum
        self.momentum = momentum

    def _create_all_weights(self, params):
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        ema_moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments + ema_moments
        return moments, ema_moments

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * tf.cast(self.iterations, K.dtype(self.decay))))

        moments, ema_moments = self._create_all_weights(params)
        for p, g, m, e in zip(params, grads, moments, ema_moments):
            # moments
            v = self.momentum * m - lr*g
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr*g
            else:
                new_p = p + v

            # weight_decay excludes bn
            if self.weight_decay and 'batch_normalization' not in p.name:
                new_p = new_p - self.weight_decay*p

            # Apply constraints and update
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))

            # EMA
            if self.ema_momentum:
                ema = self.ema_momentum * e - (1-self.ema_momentum) * new_p
            self.updates.append(K.update(e, ema))

        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'decay': float(K.get_value(self.decay)),
            'momentum': self.momentum,
            'ema_momentum': self.ema_momentum,
            'weight_decay': self.weight_decay
        }
        base_config = super(SGDW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':

    import numpy as np
    from keras.layers import Input, GlobalAveragePooling2D, Dense
    from keras.models import Model

    inpt = Input((224,224,3))
    x = GlobalAveragePooling2D()(inpt)
    x = Dense(2, activation='softmax')(x)
    model = Model(inpt, x)
    model.compile(SGDW(lr=1e-3, weight_decay=1e-4), 'categorical_crossentropy')

    X = np.random.uniform(0, 1, (32,224,224,3))
    y = np.random.random_integers(0, 1, (32,2))
    model.fit(X,y,epochs=10, batch_size=16, verbose=1)



