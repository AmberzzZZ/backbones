from keras.models import Model
from keras.layers import Layer, Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, ZeroPadding2D
from GroupConv import GroupConv2D
import keras.backend as K
import tensorflow as tf


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
                        group_dict=groups, layer_idx=layer_idx)(x)
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
            self.reparam = GroupConv2D(n_filters, kernel_size, stride=stride, groups=groups, padding='same')  # conv-bias
        else:
            # branches: bn & 3x3conv-bn & 1x1conv-bn
            if stride==1:
                self.id_branch = BatchNormalization(momentum=0.9, epsilon=1e-5)
            self.conv3_branch = ConvBN(n_filters, kernel_size, stride=stride, groups=groups)  # conv-bn-bias
            self.conv1_branch = ConvBN(n_filters, 1, stride=stride, groups=groups)

        if self.use_se:
            self.se = SEBlock(n_filters, ratio=16)

        self.relu = ReLU()

    def call(self, x):

        if self.test_mode:
            # plain
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

    # def get_equivalent_kernel_bias(self):
    #     kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
    #     kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
    #     kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
    #     return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    # def set_test_weights(self):
    #     # get merged conv & bias
    #     self.reparam.set_weights([])

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        return (b,h//self.stride,w//self.stride,self.n_filters)


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


def RepVGG_A0(input_shape=(224,224,3), num_classes=1000):
    return RepVGG(input_shape, num_classes, use_se=False, test_mode=False, groups=None,
                  num_blocks=[2,4,14,1], width_multiplier=[0.75,0.75,0.75,2.5])


def RepVGG_B1g4(input_shape=(224,224,3), num_classes=1000):
    return RepVGG(input_shape, num_classes, use_se=False, test_mode=False, groups=g4_map,
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
    print(np.argmax(probs))









