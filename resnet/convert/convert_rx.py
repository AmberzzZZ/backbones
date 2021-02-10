import h5py
from resnext import resnext
import numpy as np


if __name__ == '__main__':

    # step1: parse the pretrained weights
    f = h5py.File('../weights/resnext50_weights_tf_dim_ordering_tf_kernels.h5', 'r')
    layers = []
    for layer, g in f.items():
        if len(g.items())==0 or 'probs' in layer:
            continue
        print("layer: ", layer, len(g.items()))
        for _, param_group in g.items():
            print("number params: ", len(param_group.items()))
            if len(param_group.items())==1:   # conv, swap with bn
                layers.insert(-1, layer)
            else:
                layers.append(layer)

            # for param_name, param_weights in param_group.items():
            #     print(param_name)
            #     print(param_weights.shape)
    # print(layers)

    # step2: set weights
    my_model = resnext(input_shape=(224,224,3), depth=50)
    my_conv_layers = [i.name for i in my_model.layers if 'conv' in i.name]
    my_conv_layers = sorted(my_conv_layers, key=lambda x: int(x.split('_')[-1]))
    my_bn_layers = [i.name for i in my_model.layers if 'batch_norm' in i.name]
    my_bn_layers = sorted(my_bn_layers, key=lambda x: int(x.split('_')[-1]))
    print(len(layers), len(my_bn_layers))

    # set_weights
    print('converting....')
    C = 32
    for i in range(len(my_conv_layers)):     # for a conv-bn pair
        official_conv_layer = layers[i*2]
        official_bn_layer = layers[i*2+1]

        conv_weights = [np.array(i) for _, i in f[official_conv_layer][official_conv_layer].items()]
        conv_weights = [np.expand_dims(i, axis=0) for i in conv_weights][0]
        if conv_weights.shape[-2] == C*conv_weights.shape[-1]:
            # group conv: [N,k,k,in,out] vs.  [1,k,k,N*in,out]
            k, filters_in, filters_group = conv_weights.shape[2:]
            N = filters_in // filters_group
            conv_weights = conv_weights.reshape(k,k,N, filters_group, filters_group)
            conv_weights = np.transpose(conv_weights, (2, 0, 1, 3, 4))   # (2, 0, 1, 4, 3)
        print(conv_weights.shape)
        bn_weights = [np.array(i) for _, i in f[official_bn_layer][official_bn_layer].items()]
        print([i.shape for i in bn_weights])

        conv_index = i
        print("my layer: ", my_conv_layers[conv_index], my_bn_layers[conv_index])
        print("off layer: ", official_conv_layer, official_bn_layer)
        my_model.get_layer(name=my_conv_layers[conv_index]).set_weights([conv_weights])
        my_model.get_layer(name=my_bn_layers[conv_index]).set_weights(bn_weights)

    my_model.save_weights('rx50_outin.h5')

    f.close()











