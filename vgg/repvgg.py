from keras.models import Model
from keras.layers import Layer, Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, ZeroPadding2D
from GroupConv import GroupConv2D
import keras.backend as K
import tensorflow as tf
import numpy as np


def RepVGG(input_shape, num_classes=1000, use_se=False, test_mode=False, groups=None,
           num_blocks=[2,4,14,1], width_multiplier=[0.75,0.75,0.75,2.5]):

    start_filters = 64
    n_filters = min(int(start_filters*width_multiplier[0]), start_filters)
    if not groups:
        groups = {}

    inpt = Input(input_shape)

    # stem: single RepVGG block
    x = RepVGGBlock(n_filters, kernel_size=3, stride=2, groups=1,
                    use_se=use_se, test_mode=test_mode)(inpt)

    # stages
    layer_idx = 1
    for i in range(4):
        x = RepVGGStage(num_blocks[i], strides=2, n_filters=int(start_filters*(2**i)*width_multiplier[i]),
                        group_dict=groups, layer_idx=layer_idx, use_se=use_se, test_mode=test_mode)(x)
        layer_idx += num_blocks[i]

    # head
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes)(x)

    model = Model(inpt, x)

    return model


class RepVGGStage(Model):
    def __init__(self, n_blocks, strides, n_filters, group_dict, layer_idx=1,
                 use_se=False, test_mode=False):
        super(RepVGGStage,self).__init__()

        self.stride = strides
        self.n_filters = n_filters
        self.blocks = []

        strides = [strides] + [1]*(n_blocks-1)   # [s,1,1,...]
        for i in range(n_blocks):
            groups = group_dict.get(layer_idx, 1)
            self.tmp = RepVGGBlock(n_filters, kernel_size=3, stride=strides[i], groups=groups,
                                   use_se=use_se, test_mode=test_mode)
            self.blocks.append(self.tmp)
            layer_idx += 1

    def call(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x

    def compute_output_shape(self, input_shape):
        h, w, c = input_shape[1:]
        return (None, h//self.stride, w//self.stride, self.n_filters)

    def switch_to_deploy(self, weight_dict):

        # for k,v in weight_dict.items():
        #     print(k,v)
        print(weight_dict.keys())

        for layer in self.layers:     # blocks
            if layer.weights:
                print(layer.name)
                # print(weight_dict.keys())   # repvggblock_id
                layer.switch_to_deploy(weight_dict[layer.name])


class RepVGGBlock(Model):

    def __init__(self, n_filters, kernel_size=3, stride=1, groups=1,
                 use_se=False, test_mode=False):
        super(RepVGGBlock,self).__init__()

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_se = use_se
        self.test_mode = test_mode

        if self.test_mode:
            self.reparam = GroupConv2D(n_filters, kernel_size, strides=stride, groups=groups,
                                       padding='valid', use_bias=True)  # conv-bias
        else:
            # branches: bn & 3x3conv-bn & 1x1conv-bn
            if stride==1:
                self.id_branch = BatchNormalization(momentum=0.9, epsilon=1e-5)
            self.conv3_branch = ConvBN(n_filters, kernel_size, stride=stride, groups=groups)  # conv-bn
            self.conv1_branch = ConvBN(n_filters, 1, stride=stride, groups=groups)

        if self.use_se:
            self.se = SEBlock(n_filters, ratio=16)

        self.relu = ReLU()

    def call(self, x):

        if self.test_mode:
            # plain
            x = ZeroPadding2D(padding=self.kernel_size//2)(x)
            x = self.reparam(x)
        else:
            # branches
            if self.stride==1:
                x = self.id_branch(x) + self.conv3_branch(x) + self.conv1_branch(x)
            else:
                x = self.conv3_branch(x) + self.conv1_branch(x)

        if self.use_se:
            x = self.se(x)

        return self.relu(x)

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        return (b,h//self.stride,w//self.stride,self.n_filters)

    def switch_to_deploy(self, weight_dict):

        # single layer: group conv with bias
        # for k, v in weight_dict.items():
        #     print(k,v)
        print(weight_dict.keys())
        w = list(weight_dict.values())
        bn = None
        if len(weight_dict.keys())==3:
            bn = w[0]     # [beta, gamma, mean, var]
            bn3, conv3 = list(w[1].values())   # kernel
            bn1, conv1 = list(w[2].values())   # kernel
            # print(bn.keys(), conv3['kernel:0'].shape, conv1['kernel:0'].shape)
        else:
            bn3, conv3 = list(w[0].values())
            bn1, conv1 = list(w[1].values())

        if conv3['kernel:0'].shape[1]!=3:
            bn3, conv3, bn1, conv1 = bn1, conv1, bn3, conv3

        kernel3, bias3 = self.fuse_conv_bn(conv3, bn3)
        kernel1, bias1 = self.fuse_conv_bn(conv1, bn1, pad=1)
        reparam_kernel = kernel3 + kernel1
        reparam_bias = bias3 + bias1
        if bn is not None:
            kernel, bias = self.fuse_conv_bn(conv3, bn, only_bn=True)
            reparam_kernel += kernel
            reparam_bias += bias

        for layer in self.layers:
            if layer.weights:
                print(layer.name, np.max(reparam_kernel), np.min(reparam_kernel), np.max(reparam_bias), np.min(reparam_bias))
                layer.set_weights([reparam_kernel, reparam_bias])
                break

    def fuse_conv_bn(self, conv, bn, eps=1e-5, only_bn=False, pad=0):
        if only_bn:
            kernel = np.zeros_like(conv['kernel:0'])   # [g,k,k,g_in,g_out]
            g, k, _, g_in, g_out = kernel.shape
            for i in range(g_in):
                kernel[:,1,1,i%g_in,i] = 1
        else:
            kernel = np.array(conv['kernel:0'])   # [g,k,k,g_in,g_out]
            kernel = np.pad(kernel, [[0,0],[pad,pad],[pad,pad],[0,0],[0,0]])
        gamma = np.array(bn['gamma:0'])   # [g_out*g]
        beta = np.array(bn['beta:0'])
        mean = np.array(bn['moving_mean:0'])
        var = np.array(bn['moving_variance:0'])
        std = np.sqrt(var + eps)

        new_bias = beta - mean * gamma / std
        g, k, _, g_in, g_out = kernel.shape
        gamma = np.reshape(gamma, (g,1,1,1,g_out))   # operate on last dim
        std = np.reshape(std, (g,1,1,1,g_out))   # operate on last dim
        std = np.reshape(std, (g,1,1,1,g_out))
        new_kernel = kernel * gamma / std

        return new_kernel, new_bias


class ConvBN(Model):

    def __init__(self,n_filters, kernel_size=3, stride=1, groups=1):
        super(ConvBN, self).__init__()
        self.stride = stride
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.conv = GroupConv2D(n_filters, kernel_size, strides=stride, groups=groups, padding='valid', use_bias=False)
        self.bn = BatchNormalization(momentum=0.9, epsilon=1e-5)

    def call(self, x):
        if self.kernel_size!=1:
            pad = self.kernel_size//2
            x = ZeroPadding2D(padding=pad)(x)
        return self.bn(self.conv(x))

    def compute_output_shape(self, input_shape):
        h, w, c = input_shape[1:]
        return (None, h//self.stride, w//self.stride, self.n_filters)


class SEBlock(Model):

    def __init__(self, n_filters, ratio=16):
        super(SEBlock, self).__init__()
        self.n_filters = n_filters
        self.conv_down = Conv2D(n_filters//ratio, 1, stride=1, padding='same', activation='relu')
        self.conv_up = Conv2D(n_filters, 1, stride=1, padding='same', activation='sigmoid')

    def call(self, x):
        inpt = x        # [b,h,w,c]
        # gap
        x = GlobalAveragePooling2D(keepdims=True)(x)
        # narrow the channel dim
        x = self.conv_down()(x)
        # recover the channel dim
        x = self.conv_up()(x)     # [b,1,1,c]
        # reweight
        x = inpt * x
        return x

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        return (b,h,w,self.n_filters)


########## model zoo ###########
optional_groupwise_layers = [2,4,6,8,10,12,14,16,18,20,22,24,26]
g2_map = {l:2 for l in optional_groupwise_layers}   # layer_idx:
g4_map = {l:4 for l in optional_groupwise_layers}


def RepVGG_A0(input_shape=(224,224,3), num_classes=1000, test_mode=False):
    return RepVGG(input_shape, num_classes, use_se=False, test_mode=test_mode, groups=None,
                  num_blocks=[2,4,14,1], width_multiplier=[0.75,0.75,0.75,2.5])


def RepVGG_B1g4(input_shape=(224,224,3), num_classes=1000, test_mode=False):
    return RepVGG(input_shape, num_classes, use_se=False, test_mode=test_mode, groups=g4_map,
                  num_blocks=[4,6,16,1], width_multiplier=[2,2,2,4])


if __name__ == '__main__':

    model = RepVGG_B1g4((224,224,3))
    model.summary()
    model.load_weights("weights/RepVGG-B1g4-train.h5")

    import cv2
    import numpy as np

    img = cv2.imread("/Users/amber/Downloads/cat.jpeg", 1)
    img = cv2.resize(img, (224,224))
    img = img / 255.
    img = np.expand_dims(img, axis=0)

    probs = model.predict(img)[0]
    print(np.argmax(probs), np.max(probs))









