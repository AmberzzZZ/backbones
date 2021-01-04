from keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D, MaxPooling2D, add
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import Model
from hourglass import Conv_BN, residual


def hourglass(input_shape=(512,512,3), n_classes=80, n_stacks=2, n_channles=[256, 384, 384, 384, 512]):
    inpt = Input(input_shape)

    # stem: 7x7 s2 conv + s2 residual
    x = Conv_BN(inpt, 128, 3, strides=2, activation='relu')
    x = residual(x, 256, strides=2)

    # hourglass modules
    outputs = []
    for i in range(n_stacks):
        x, x_intermediate = hourglass_module(x, n_classes, n_channles)
    outputs.append(x_intermediate)

    # model
    model = Model(inpt, outputs)
    model.compile(Adam(lr=2.5e-4), loss=mean_squared_error, metrics=['acc'])

    return model


def hourglass_module(inpt, n_classes, n_channles):
    x = inpt
    features = [x]     # from x4 to x64
    n_levels = len(n_channles)
    # encoder: s1 residual + s2 residual
    for i in range(n_levels):
        if i!=0:
            x = residual(x, n_channles[i], strides=1)
            features.append(x)
        if i!=n_levels-1:
            x = residual(x, n_channles[i+1], strides=2)

    # mid connection: 4 residuals
    for i in range(4):
        x = residual(x, n_channles[-1], strides=1)

    # decoder:
    for i in range(n_levels-2, -1, -1):
        # skip: 2 residuals
        skip = features[i]
        skip = residual(skip, n_channles[i], strides=1)
        skip = residual(skip, n_channles[i], strides=1)
        # features
        x = residual(x, n_channles[i])
        x = residual(x, n_channles[i])
        x = UpSampling2D()(x)
        # add
        x = add([x, skip])

    # head branches
    x_intermediate = Conv2D(n_classes, 1, strides=1, padding='same')(x)
    input_branch = Conv2D(n_channles[0], 1, strides=1, padding='same')(inpt)
    output_branch = Conv2D(n_channles[0], 1, strides=1, padding='same')(x)
    x = add([input_branch, output_branch])
    x = ReLU()(x)

    return x, x_intermediate


if __name__ == '__main__':

    model = hourglass(n_stacks=1, n_channles=[256, 384])
    # model.summary()
    # model.save("hourglass_corner.h5")










