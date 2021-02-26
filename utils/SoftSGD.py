from keras.optimizers import Optimizer
import keras.backend as K


class SoftSGD(Optimizer):
    # [new arg] steps_per_update: how many batch to update gradient
    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, steps_per_update=2, **kwargs):
        super(SoftSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.steps_per_update = steps_per_update  # 多少batch才更新一次
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov

    def get_updates(self, loss, params):
        # learning rate decay
        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        shapes = [K.int_shape(p) for p in params]
        sum_grads = [K.zeros(shape) for shape in shapes]  # 平均梯度，用来梯度下降
        grads = self.get_gradients(loss, params)  # 当前batch梯度
        self.updates = [K.update_add(self.iterations, 1)]
        self.weights = [self.iterations] + sum_grads
        for p, g, sg in zip(params, grads, sum_grads):
            # momentum 梯度下降
            v = self.momentum * sg / float(self.steps_per_update) - lr * g  # velocity
            if self.nesterov:
                new_p = p + self.momentum * v - lr * sg / float(self.steps_per_update)
            else:
                new_p = p + v

            # constraints on params
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            # update grads on certain condition
            cond = K.equal(self.iterations % self.steps_per_update, 0)
            self.updates.append(K.switch(cond, K.update(p, new_p), p))
            self.updates.append(K.switch(cond, K.update(sg, g), K.update(sg, sg + g)))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'steps_per_update': self.steps_per_update,
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov
                  }
        base_config = super(SoftSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


