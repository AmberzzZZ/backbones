from keras.optimizers import Optimizer
import keras.backend as K
import tensorflow as tf


class AdamW(Optimizer):
    # Adam optimizer with weight decay

    def __init__(self,
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 weight_decay=0.,
                 epsilon=None,
                 decay=0.,
                 amsgrad=False,
                 **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def _create_all_weights(self, params):
        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats
        return ms, vs, vhats

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * tf.cast(self.iterations, K.dtype(self.decay))))

        with tf.control_dependencies([tf.compat.v1.assign_add(self.iterations, 1)]):
            t = tf.cast(self.iterations, K.floatx())
        # ERM approx correction
        lr_t = lr * (K.sqrt(1. - tf.pow(self.beta_2, t)) / (1. - tf.pow(self.beta_1, t)))

        ms, vs, vhats = self._create_all_weights(params)
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * tf.square(g)
            if self.amsgrad:
                vhat_t = tf.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(tf.compat.v1.assign(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(tf.compat.v1.assign(m, m_t))
            self.updates.append(tf.compat.v1.assign(v, v_t))

            # weight decay
            new_p = p_t - p*self.weight_decay

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(tf.compat.v1.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'weight_decay': float(K.get_value(self.weight_decay)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
        }
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





