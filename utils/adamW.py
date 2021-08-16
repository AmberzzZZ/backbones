from keras.optimizers import Optimizer
import keras.backend as K
import tensorflow as tf


class AdamW(Optimizer):
    # Adam optimizer with weight decay & ema

    def __init__(self,
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 decay=0.,
                 amsgrad=False,
                 weight_decay=4e-5,
                 ema_momentum=0.9999,
                 **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.weight_decay = weight_decay
        self.ema_momentum = ema_momentum

    def _create_all_weights(self, params):
        ms = [K.zeros(K.int_shape(p), name=p.name.split('/')[0]+'/ms') for p in params]
        vs = [K.zeros(K.int_shape(p), name=p.name.split('/')[0]+'/vs') for p in params]
        ema_weights = [tf.Variable(p, name=p.name.split('/')[0]+'/ema') for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), name=p.name.split('/')[0]+'/vhats') for p in params]
        else:
            vhats = [K.zeros(1, name=p.name.split('/')[0]+'/vhats') for p in params]
        self.weights = [self.iterations] + ms + vs + vhats + ema_weights
        return ms, vs, vhats, ema_weights

    def get_updates(self, loss, params):
        # old weights: params
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * tf.cast(self.iterations, K.dtype(self.decay))))

        # EMA bias correction
        t = tf.cast(self.iterations+1, tf.float32)
        lr_t = lr * (K.sqrt(1. - tf.pow(self.beta_2, t)) / (1. - tf.pow(self.beta_1, t)))
        # self.updates.append(K.update(self.lr, lr_t))

        ms, vs, vhats, ema_weights = self._create_all_weights(params)
        for p, g, m, v, vhat, e in zip(params, grads, ms, vs, vhats, ema_weights):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * tf.square(g)
            if self.amsgrad:
                vhat_t = tf.maximum(vhat, v_t)
                new_p = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                new_p = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            # weight decay
            if self.weight_decay:
                new_p = new_p - p*self.weight_decay*lr

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))

            # EMA
            if self.ema_momentum:
                ema = self.ema_momentum * e - (1.-self.ema_momentum)*new_p
                self.updates.append(K.update(e, ema))

        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': self.weight_decay,
            'ema_momentum': self.ema_momentum,
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
        }
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_ema_weights(self):
        ema_weights = [i for i in self.weights if 'ema' in i.name]
        return ema_weights


if __name__ == '__main__':

    import numpy as np
    from keras.layers import Input, GlobalAveragePooling2D, Dense
    from keras.models import Model, load_model
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adam


    inpt = Input((None,None,3))
    x = GlobalAveragePooling2D()(inpt)
    x = Dense(2, activation='softmax', kernel_initializer='zeros', bias_initializer='zeros')(x)
    model = Model(inpt, x)
    optimizer = AdamW()
    # optimizer = Adam()
    model.compile(optimizer, 'categorical_crossentropy')

    # train
    # X = np.random.uniform(0, 1, (32,224,224,3))
    # Y = np.random.randint(0, 2, (32,2))
    X = np.load("X.npy")
    Y = np.load("Y.npy")
    ckpt = ModelCheckpoint('test_ep{epoch:02d}.h5', monitor='loss', verbose=1)
    model.fit(X,Y,epochs=10, batch_size=20, verbose=1, callbacks=[])

    # # test
    # model = load_model('test_ep01.h5', custom_objects={'AdamW': AdamW})
    # ema_weights = K.batch_get_value(model.optimizer.get_ema_weights())
    # old_weights = K.batch_get_value(model.weights)
    # print(len(ema_weights), len(old_weights))
    # print("old weights")
    # for layer in model.layers:
    #     print(layer.name)
    #     print([np.array(i) for i in layer.get_weights()])
    # print("ema weights")
    # K.batch_set_value(zip(model.weights, ema_weights))
    # for layer in model.layers:
    #     print(layer.name)
    #     print([np.array(i) for i in layer.get_weights()])
    # print("reset to old weights")
    # K.batch_set_value(zip(model.weights, old_weights))
    # for layer in model.layers:
    #     print(layer.name)
    #     print([np.array(i) for i in layer.get_weights()])






