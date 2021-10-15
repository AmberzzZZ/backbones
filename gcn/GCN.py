from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import keras.backend as K
import time
import numpy as np
from GraphConvolution import GraphConvolution
from dataLoader import load_data, get_splits, preprocess_adj, eval_preds


def GCN(n_classes=7):

    inpt = Input((2708,1433))      # (b,N,D)
    adj = Input((2708,2708))       # (b,N,N)

    x = Dropout(0.5)(inpt)
    x = GraphConvolution(16, activation='relu')([x,adj])      # (b,N,F)
    x = Dropout(0.5)(x)
    x = GraphConvolution(n_classes, activation='softmax')([x,adj])     # (b,N,cls)

    model = Model([inpt,adj], x)

    return model


def prepare_data(X, A, *args):

    X /= X.sum(1)
    X = np.expand_dims(X, axis=0)
    A = preprocess_adj(A, symmetric=True).todense()
    A = np.expand_dims(A, axis=0)   # repeat batch-dim if needed
    args = [np.expand_dims(a, axis=0) for a in args]

    return X, A, args


if __name__ == '__main__':

    # data
    X, A, y = load_data()       # (N,D)mat, (N,N)sp_mat, (N,cls)mat
    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)
    X, A, y_lst = prepare_data(X, A, y, train_mask)
    y, train_mask = y_lst
    n_classes = y_train.shape[-1]

    def ce_on_train(y_true, y_pred):
        mask = K.constant(train_mask, dtype='float32')   # (b,N)
        ce = categorical_crossentropy(y_true, y_pred)    # (b,N,cls)
        return ce * mask

    # model
    model = GCN(n_classes=n_classes)
    model.compile(optimizer=Adam(lr=0.01), loss=ce_on_train)

    # train
    print("############### train #################")
    best_acc = 0.
    best_loss = 9999999
    wait = 0
    for ep in range(1, 201):

        t = time.time()

        # single train
        model.fit([X,A],y,
                  batch_size=1,
                  epochs=1,
                  shuffle=False,
                  verbose=1)

        # single validation
        preds = model.predict([X,A], batch_size=1)[0]     # (N,cls)

        train_val_loss, train_val_acc = eval_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])

        # log
        print("Epoch: {:04d}".format(ep),
              "train_loss: {:.3f}".format(train_val_loss[0]),
              "train_acc: {:.3f}".format(train_val_loss[1]),
              "test_loss: {:.3f}".format(train_val_acc[0]),
              "test_acc: {:.3f}".format(train_val_acc[1]),
              "time: {:.4f}".format(time.time()-t),
              )

        # save ckpt
        if train_val_acc[1]>best_acc or train_val_loss[1]<best_loss:
            best_acc = max(best_acc, train_val_acc[1])
            best_loss = min(best_loss, train_val_loss[1])
            model.save_weights("GCN_cls%d_ep%d_valacc_%f.h5" % (n_classes, ep, best_acc))
            wait = 0
        else:
            if wait>10:
                break
            wait += 1

    # test
    print("############### test #################")
    test_loss, test_acc = eval_preds(preds, [y_test], [idx_test])
    print("test_loss: {:.3f}".format(test_loss[0]),
          "test_acc: {:.3f}".format(test_acc[0]))



    # ############### test #################
    # test_loss: 0.629 test_acc: 0.808




