from keras.layers import Conv2D, Input
from keras.utils import conv_utils
from keras.engine import InputSpec
import keras.backend as K
import tensorflow as tf
from keras.models import Model


class GroupConv2D(Conv2D):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 group=1,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(GroupConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.channel_axis = None
        self.group = group

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs to `groupConv2d` should have rank 4. '
                             'Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = -1
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        if input_dim % self.group != 0:
            raise ValueError('input channel num should be '
                             'integral multiple of group_num.')

        if self.filters % self.group != 0:
            raise ValueError('output channel num should be '
                             'integral multiple of group_num.')

        group_size = input_dim // self.group
        group_filters = self.filters // self.group
        kernel_shape = (self.group,) + self.kernel_size + (group_size, group_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.group * group_filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        else:
            self.bias = None


        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={self.channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # split
        group_inputs = tf.split(inputs, self.group, axis=self.channel_axis)
        # tf.nn.conv2d
        group_outputs = [K.conv2d(inp, self.kernel[i,...],
                                  strides=self.strides, padding=self.padding,
                                  dilation_rate=self.dilation_rate, data_format=self.data_format)
                                  for i, inp in enumerate(group_inputs)]
        # concat
        outputs = tf.concat(group_outputs, axis=self.channel_axis)

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            output_h = int(input_shape[2] / self.strides[0])
            output_w = int(input_shape[3] / self.strides[1])
            return input_shape[0], self.filters, output_h, output_w

        elif self.data_format == 'channels_last':
            output_h = int(input_shape[1] / self.strides[0])
            output_w = int(input_shape[2] / self.strides[1])
            return input_shape[0], output_h, output_w, self.filters


if __name__ == '__main__':

    x = Input((224,224,3))
    y = GroupConv2D(64, 7, strides=2, padding='same', group=1)(x)
    model = Model(x,y)
    model.summary()









