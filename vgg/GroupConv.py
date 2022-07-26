from keras.layers import Conv2D, Input
from keras.engine import InputSpec
import keras.backend as K
import tensorflow as tf
from keras.models import Model


class GroupConv2DKT(Conv2D):
    # keras-team's implementation of group conv
    # 0. with feature input: [h,w,128], C=32, g_filters=4
    # 1. depthwise conv: depth_multiplier=g_filters, [h,w,128,4]
    # 2. reshape: splitting groups, [h,w,32,4,4]
    # 3. fuse: sum by groups, by axis-1, [h,w,32,4]
    # 4. reshape: flatten back to feature dims, [h,w,128]
    # 5. add_bias

    def __init__(self, filters, kernel_size, strides=1, groups=1,
                 use_bias=False,
                 depthwise_initializer='glorot_uniform',
                 depthwise_regularizer=None,
                 depthwise_constraint=None,
                 *args, **kwargs):
        super(GroupConv2DKT, self).__init__(filters=filters,
                                            kernel_size=kernel_size,
                                            *args, **kwargs)
        self.strides = strides
        self.groups = groups
        self.depthwise_initializer = depthwise_initializer
        self.depthwise_regularizer = depthwise_regularizer
        self.depthwise_constraint = depthwise_constraint
        self.use_bias = use_bias

        self.channel_axis = -1

    def build(self, input_shape):
        input_dim = input_shape[self.channel_axis]
        assert input_dim % self.groups==0 and len(input_shape)==4 and self.filters % self.groups==0

        self.group_in = input_dim // self.groups
        self.group_out = self.filters // self.groups

        self.kernel = self.add_weight(shape=self.kernel_size+(input_dim, self.group_out),
                                      name='kernel',
                                      initializer=self.depthwise_initializer,
                                      regularizer=self.depthwise_regularizer,
                                      constraint=self.depthwise_constraint,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, input):

        # input: [b,h,w,c_in]
        # depthwise: [b,h,w,c_in*group_out]
        padding_pattern = {'same': 'SAME', 'valid': 'VALID'}
        dim_pattern = {'channels_last': 'NHWC', 'channels_first': 'NCHW'}
        self.strides = (1,) + self.strides + (1,)

        x = tf.nn.depthwise_conv2d(input, self.kernel,
                                   strides=self.strides, padding=padding_pattern[self.padding],
                                   data_format=dim_pattern[self.data_format])
        # reshape: [b,h,w,groups, group_out, group_in]
        h, w = K.int_shape(x)[1:3]
        x = tf.reshape(x, (-1, h, w, self.groups, self.group_out, self.group_in))
        # fuse: [b,h,w,groups,group_out]
        x = tf.reduce_sum(x, axis=-1)
        # reshape: [b,h,w,filters]
        x = tf.reshape(x, (-1, h, w, self.filters))
        # add bias: [b,h,w,filters]
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format=dim_pattern[self.data_format])
        if self.activation:
            return self.activation(x)
        return x

    def compute_output_shape(self, input_shape):
        # [b,h,w,cout]
        output_h = int(input_shape[1] / self.strides[1])
        output_w = int(input_shape[2] / self.strides[2])
        return input_shape[0], output_h, output_w, self.filters


class GroupConv2D(Conv2D):

    def __init__(self, filters, kernel_size, strides=1, groups=4, *args, **kwargs):
        super(GroupConv2D, self).__init__(filters=filters,
                                          kernel_size=kernel_size,
                                          *args, **kwargs)
        self.strides = strides
        self.groups = groups
        self.channel_axis = -1

    def build(self, input_shape):
        input_dim = input_shape[self.channel_axis]
        assert input_dim % self.groups==0 and len(input_shape)==4 and self.filters % self.groups==0

        group_in = input_dim // self.groups
        group_out = self.filters // self.groups

        self.kernel = self.add_weight(shape=(self.groups,)+self.kernel_size+(group_in, group_out),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.groups * group_out,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        # split
        groups_inputs = tf.split(inputs, self.groups, axis=self.channel_axis)
        # tf.nn.conv2d
        if isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)
        groups_outputs = [K.conv2d(inp, self.kernel[i,...],
                                   strides=self.strides, padding=self.padding,
                                   dilation_rate=self.dilation_rate, data_format=self.data_format)
                                   for i, inp in enumerate(groups_inputs)]
        # concat
        outputs = tf.concat(groups_outputs, axis=self.channel_axis)

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        # [b,h,w,cout]
        output_h = int(input_shape[1] / self.strides[0])
        output_w = int(input_shape[2] / self.strides[1])
        return input_shape[0], output_h, output_w, self.filters


if __name__ == '__main__':

    x = Input((56,56,16))

    y = GroupConv2D(64, 7, strides=2, padding='same', use_bias=False, groups=1)(x)
    model = Model(x,y)
    model.summary()

    layer = GroupConv2D(64, 7, strides=2, padding='same', use_bias=False, groups=1)
    print(layer.weights)

    # y = GroupConv2DKT(64, 7, strides=2, padding='same', use_bias=False, groups=1)(x)
    # model = Model(x,y)
    # model.summary()
