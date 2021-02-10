import numpy as np
import configparser
from collections import defaultdict
import io
from keras.layers import Input, Conv2D, ZeroPadding2D, Add, UpSampling2D, MaxPooling2D, Concatenate, BatchNormalization, LeakyReLU, Lambda, GlobalAveragePooling2D, Reshape
from keras import backend as K
from keras.regularizers import l2
import tensorflow as tf
from keras.models import Model


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.
    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


if __name__ == '__main__':

    config_pt = "weights/csp_r50.cfg"
    weight_pt = "weights/csresnet50.weights"

    weights_file = open(weight_pt, 'rb')
    config_file = unique_config_sections(config_pt)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(config_file)
    major, minor, revision = np.ndarray(
        shape=(3, ), dtype='int32', buffer=weights_file.read(12))
    if (major*10+minor)>=2 and major<1000 and minor<1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    # print(cfg_parser.sections())   # list of layer
    # # for key in cfg_parser['net_0']:    # net hyperparameters
    # #     print(key)
    # # for key in cfg_parser['convolutional_0']:    # bn,filters,size,stride,pad,activation
    # #     print(key)

    print('Creating Keras model.')
    input_layer = Input(shape=(256, 256, 3))
    prev_layer = input_layer
    all_layers = []

    weight_decay = float(cfg_parser['net_0']['decay']
                         ) if 'net_0' in cfg_parser.sections() else 5e-4
    count = 0
    out_index = []
    for section in cfg_parser.sections()[:]:
        print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            padding = 'same'
            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)    # [B,H,W,C]

            weights_shape = (size, size, prev_layer_shape[-1], filters)      # tf [h, w, in, out]
            darknet_w_shape = (filters, prev_layer_shape[-1], size, size)    # dark [out, in, h, w]
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn' if batch_normalize else '  ', activation, weights_shape)

            conv_bias = np.ndarray(
                shape=(filters, ),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            count += filters

            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [conv_weights, conv_bias]

            # Handle activation.
            act_fn = None
            if activation == 'leaky':
                pass  # Add advanced activation later.
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(activation, section))

            print("number of conv weights: ", len(conv_weights))
            # Create Conv2D layer
            # if stride>1:
            #     # Darknet uses left and top padding instead of 'same' mode
            #     prev_layer = ZeroPadding2D(((1,0),(1,0)))(prev_layer)
            conv_layer = (Conv2D(
                filters, (size, size),
                strides=(stride, stride),
                kernel_regularizer=l2(weight_decay),
                use_bias=not batch_normalize,
                weights=conv_weights,
                activation=act_fn,
                padding=padding))(prev_layer)
            if batch_normalize:
                conv_layer = (BatchNormalization(weights=bn_weight_list))(conv_layer)
            prev_layer = conv_layer

            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                print('Concatenating route layers:', layers)
                concatenate_layer = Concatenate()(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                skip_layer = layers[0]  # only one layer to route
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                MaxPooling2D(
                    pool_size=(size, size),
                    strides=(stride, stride),
                    padding='same')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'):
            index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            # assert activation == 'linear', 'Only linear activation supported.'
            if K.int_shape(all_layers[index])[-1]==K.int_shape(prev_layer)[-1]:
                act_layer = Add()([all_layers[index], prev_layer])
            else:
                # pad
                gap = K.int_shape(prev_layer)[-1] - K.int_shape(all_layers[index])[-1]
                if gap>0:
                    # pad previous
                    act_layer = Lambda(lambda x: tf.pad(x, [[0,0],[0,0],[0,0],[0,gap]]))(all_layers[index])
                    act_layer = Add()([act_layer, prev_layer])
                else:
                    # pad current
                    act_layer = Lambda(lambda x: tf.pad(x, [[0,0],[0,0],[0,0],[0,gap]]))(prev_layer)
                    act_layer = Add()([all_layers[index], act_layer])
            if activation != 'linear':
                print("shortcut activation: ", activation)
                act_layer = LeakyReLU(alpha=0.1)(act_layer)
            all_layers.append(act_layer)
            prev_layer = all_layers[-1]

        elif section.startswith('net'):
            pass

        elif section.startswith('avgpool'):
            all_layers.append(GlobalAveragePooling2D()(prev_layer))
            prev_layer = all_layers[-1]
            all_layers.append(Reshape((1,1,1024))(prev_layer))
            prev_layer = all_layers[-1]

        else:
            pass
            # raise ValueError(
            #     'Unsupported section header type: {}'.format(section))

    # save model
    model = Model(inputs=input_layer, outputs=prev_layer)
    model.summary()
    # model.save('csp_r50.h5')
    model.save_weights('csp_r50.h5')


