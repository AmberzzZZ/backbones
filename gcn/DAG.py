from keras.layers import Input, Concatenate, Lambda, Dense, ReLU, add
from keras.models import Model
import keras.backend as K
from GraphConvolution import ResGraphConvolution


def GCN_global(in_dim=256, out_dim=256, out_features=9, num_nodes=24):

    inpt = Input((num_nodes, in_dim))   # (b,N,D)
    adj = Input((num_nodes, num_nodes))   # (b,N,N)
    delta = Input((num_nodes, num_nodes*2))       # (b,N,N*2)

    # GCN feature
    x = ResGraphConvolution(out_dim, activation=None, use_bias=True)(inpt,adj)
    id = x
    x = ResGraphConvolution(out_dim, activation=None, use_bias=True)(x,adj)
    x = ResGraphConvolution(out_dim, activation=None, use_bias=True)(x,adj)
    x = add([id, x])

    # fuse structral feature
    x = Concatenate(axis=-1)([x, delta])    # (b,N,256+48)

    # narrow dim
    x = ResGraphConvolution(32, activation=None, use_bias=True)(x,adj)   # (b,N,32)
    # output
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False))(x)      # (b,32)
    x = Dense(out_features, activation=None, use_bias=True)(x)

    model = Model([inpt, adj, delta], x)

    return model


def GCN_local(in_dim=256, out_dim=256, out_features=2, num_nodes=24):

    inpt = Input((num_nodes, in_dim))   # (b,N,D)
    adj = Input((num_nodes, num_nodes))   # (b,N,N)
    delta = Input((num_nodes, num_nodes*2))       # (b,N,N*2)

    # GCN feature
    x = ResGraphConvolution(out_dim, activation=None, use_bias=True)(inpt,adj)
    id = x
    x = ResGraphConvolution(out_dim, activation=None, use_bias=True)(x,adj)
    x = ResGraphConvolution(out_dim, activation=None, use_bias=True)(x,adj)
    x = add([id, x])
    x = ReLU()(x)

    # fuse structral feature
    x = Concatenate(axis=-1)([x, delta])    # (b,N,256+48)

    # narrow dim
    x = ResGraphConvolution(32, activation=None, use_bias=True)(x,adj)   # (b,N,32)
    # output
    x = Dense(out_features, activation=None, use_bias=True)(x)   # (b,N,2)

    model = Model([inpt, adj, delta], x)

    return model















