from keras.layers import Input, Conv2D, BatchNormalization, Activation, add, UpSampling2D, \
                         concatenate, GlobalAveragePooling2D, Dense
from keras.models import Model
import keras.backend as K
from config import hrnet18, hrnet32, hrnet48


BN_MOMENTUM = 0.1


def HRNet(input_shape, n_classes=1000, stem_filters=64, cfg=hrnet18):

    inpt = Input(input_shape)

    # stem stage: 4 bottleneck residial
    x = ConvBN(inpt, stem_filters, kernel_size=3, strides=2, activation='relu')
    x = ConvBN(x, stem_filters, kernel_size=3, strides=2, activation='relu')

    # stage1: 4 bottleneck residuals
    for i in range(4):
        x = residual_bn(x, stem_filters, 3, strides=1)
    n_filters = cfg[0]['n_filters'][0]
    x_align = ConvBN(x, n_filters, kernel_size=3, strides=1, activation='relu')   # align conv
    x_new = ConvBN(x_align, n_filters*2, kernel_size=3, strides=2, activation='relu')  # branch conv
    x_list = [x_align, x_new]

    # stage234: exchange blocks
    for s in range(3):        # tranverse stage
        opts = cfg[s]
        for b in range(opts['n_blocks']):     # tranverse block
            if b==opts['n_blocks']-1 and s!=2:
                # last block but not last stage, add new resolution
                x_list = exchange_block(x_list, opts, cross_stage=True)
            else:
                x_list = exchange_block(x_list, opts, cross_stage=False)

    # make head
    x = make_cls_head(x_list, opts['n_filters'], n_classes)
    # x = make_seg_head(x_list, opts['n_filters'], n_classes)
    model = Model(inpt, x)

    return model


def make_cls_head(inputs, n_filters, n_classes):
    # downSamp & sum
    x = residual_bn(inputs[0], n_filters[0]*4, kernel_size=3, strides=1)
    for i in range(1,len(n_filters)):
        # downSamp: s2-3x3-conv-bn-relu
        x = ConvBN(x, n_filters[i]*4, kernel_size=3, strides=2, activation='relu')
        # wider: bottleneck
        y = residual_bn(inputs[i], n_filters[i]*4, kernel_size=3, strides=1)
        # fuse: add
        x = add([x,y])

    x = ConvBN(x, 2048, kernel_size=1, strides=1, activation='relu')

    # predict logits
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, activation=None, use_bias=True)(x)

    return x


def make_seg_head(inputs, n_filters, n_classes):
    # upSamp & concat
    concat_lst = []
    for i, inpt in enumerate(inputs):
        if i==0:
            concat_lst.append(inpt)
        else:
            concat_lst.append(UpSampling2D(size=2**i, interpolation='nearest')(inpt))
    x = concatenate(concat_lst, axis=-1)

    # predict heatmap: conv-bn-relu-conv
    x = ConvBN(x, sum(n_filters), 1, strides=1, activation='relu')
    x = Conv2D(n_classes, 1, strides=1, activation=None, use_bias=True)(x)

    return x


def exchange_block(inputs, opts, cross_stage=False):
    # multi-resolution in parallel
    n_residuals = opts['n_residuals']
    n_filters = opts['n_filters']
    n_paths = len(n_filters)

    outputs = []
    for i in range(n_paths):
        # several basic residuals
        x = inputs[i]
        for j in range(n_residuals):
            x = residual_basic(x, n_filters[i], kernel_size=3, strides=1)
        outputs.append(x)

    # one exchange unit
    outputs = exchange_unit(outputs, opts, cross_stage)

    return outputs


def exchange_unit(inputs, opts, cross_stage=False):
    n_filters = opts['n_filters']    # for each level
    n_paths = len(n_filters)
    outputs = []
    for i in range(n_paths):      # tranverse outputs y
        add_list = []
        for j in range(n_paths):    # tranverse inputs x
            if i==j:
                # identity
                add_list.append(inputs[j])
            elif i<j:
                # upSamp
                x = ConvBN(inputs[j], n_filters[i], 1, strides=1, activation=None)
                x = UpSampling2D(size=2**(j-i), data_format=None, interpolation='nearest')(x)
                add_list.append(x)
            else:
                # downSamp
                x = inputs[j]
                for k in range(i-j-1):
                    # conv-bn-relu
                    x = ConvBN(x, n_filters[i], 3, strides=2, activation='relu')
                # conv-bn
                x = ConvBN(x, n_filters[i], 3, strides=2, activation=None)
                add_list.append(x)
        y = Activation('relu')(add(add_list))
        outputs.append(y)
    if cross_stage:
        # conv-bn-relu
        outputs.append(ConvBN(y, n_filters[-1]*2, 3, strides=2, activation='relu'))
    return outputs


def residual_bn(inpt, n_filters, kernel_size=3, strides=1):
    # 3x conv: 1-3-1
    x = inpt
    x = ConvBN(x, n_filters//4, kernel_size=1, strides=strides, activation='relu')
    x = ConvBN(x, n_filters//4, kernel_size, strides=1, activation='relu')
    x = ConvBN(x, n_filters, kernel_size=1, strides=1, activation=None)

    # id path
    if strides>1 or n_filters!=K.int_shape(inpt)[-1]:
        inpt = ConvBN(inpt, n_filters, kernel_size=1, strides=strides, activation=None)

    x = add([inpt, x])
    return Activation('relu')(x)


def residual_basic(inpt, n_filters, kernel_size=3, strides=1):
    # 2x conv: 3-3
    x = inpt
    x = ConvBN(x, n_filters, kernel_size, strides=strides, activation='relu')
    x = ConvBN(x, n_filters, kernel_size, strides=1, activation=None)

    # id path
    if strides>1 or n_filters!=K.int_shape(inpt)[-1]:
        inpt = ConvBN(inpt, n_filters, kernel_size=1, strides=strides, activation=None)

    x = add([inpt, x])
    return Activation('relu')(x)


def ConvBN(inpt, n_filters, kernel_size=3, strides=1, padding='same', activation='relu'):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding=padding, use_bias=False,
               kernel_initializer='glorot_uniform')(inpt)
    x = BatchNormalization(axis=-1, momentum=BN_MOMENTUM)(x)
    if not activation:
        x = Activation(activation)(x)
    return x


if __name__ == '__main__':

    hrnet = HRNet((512,512,3), stem_filters=64, cfg=hrnet32)
    hrnet.summary()









