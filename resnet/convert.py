import h5py
from resnet import resnet
from resnet_bias import resnet_bias
from keras.models import Model
import numpy as np


if __name__ == '__main__':

    # # step1: save h5 file by my model name
    # model = resnet_bias(input_shape=(224,224,3), depth=50, pooling=True)
    # model.load_weights("weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

    # new_model = Model(model.inputs, model.get_layer(index=-2).output)
    # new_model.save_weights("weights/r50_withbias.h5")

    # step2: read conv1 weights, duplicate across channel
    f = h5py.File('weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')
    # read conv1 weights
    for layer, g in f.items():
        for name, weights in g.items():
            if name == 'conv1_W:0':
                print('name:   ', name, weights.shape)
                print(type(weights))
                print(weights.shape, np.max(weights), np.min(weights))    # (7, 7, 3, 64) 0.70432377 -0.6710244
                tmp = weights[:]
                print(type(tmp))
    f.close()
    # # check
    # f = h5py.File('weights/r50_withbias.h5', 'r')
    # # read conv1 weights
    # for layer, g in f.items():
    #     for name, weights in g.items():
    #         if name == 'conv2d_1':
    #             weights = weights.get('kernel:0')
    #             print('name:   ', name, weights.shape)
    #             print(type(weights))
    #             print(weights.shape, np.max(weights), np.min(weights))
    # f.close()

    # step3: set weights
    ch1_model = resnet(input_shape=(224,224,1), depth=50)
    ch3_model = resnet_bias(input_shape=(224,224,3), depth=50)
    ch3_model.load_weights("weights/r50_withbias.h5")
    # set_weights
    print('converting....')
    ch1_model.get_layer(index=1).set_weights([tmp[:,:,1:2,:]])
    for i in range(2, len(ch3_model.layers)):
        print('layer ', ch1_model.get_layer(index=i).name)
        if 'conv' in ch1_model.get_layer(index=i).name:
            ch1_model.get_layer(index=i).set_weights(ch3_model.get_layer(index=i).get_weights()[:1])
        else:
            ch1_model.get_layer(index=i).set_weights(ch3_model.get_layer(index=i).get_weights())
    ch1_model.save_weights('weights/r50_ch1.h5')

    # step4: check
    f = h5py.File('weights/r50_ch1.h5', 'r')
    # read conv1 weights
    for layer, g in f.items():
        for name, weights in g.items():
            if name == 'conv2d_1':
                weights = weights.get('kernel:0')
                print(weights.shape, np.max(weights), np.min(weights))
    f.close()









