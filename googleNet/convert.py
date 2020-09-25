import h5py
import numpy as np
from efficientNet import *


if __name__ == '__main__':

    f = h5py.File('eff_b4_notop.h5', 'r')

    # read conv1 weights
    for layer, g in f.items():
        for name, d in g.items():
            if name == 'conv2d_1':
                print('name:   ', name, d)
                weights = d.get('kernel:0')
                print(type(weights))
                print(weights.shape, np.max(weights), np.min(weights))
                tmp = weights[:]
                print(type(tmp))
    f.close()

    # pad to (3,3,5,48)
    weights = np.pad(tmp, ((0,0),(0,0),(1,1),(0,0)), mode='constant')

    # ch3_model
    ch3_model = EfficientNetB4()
    ch3_model.load_weights("/Users/amber/Downloads/Misc/efficientnet-b4_weights_tf_dim_ordering_tf_kernels.h5", by_name=True, skip_mismatch=True)

    # ch5_model
    ch5_model = EfficientNet((380,380,5), 1.4, 1.8, 0.4)

    # set_weights
    print('converting....')
    ch5_model.get_layer(index=1).set_weights([weights])
    for i in range(2, len(ch3_model.layers)):
        print('layer ', ch5_model.get_layer(index=i).name)
        ch5_model.get_layer(index=i).set_weights(ch3_model.get_layer(index=i).get_weights())

    ch5_model.summary()
    # ch5_model.save_weights('effb4_ch5.h5')

