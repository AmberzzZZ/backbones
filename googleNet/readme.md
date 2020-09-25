## deeplab的xception相比原论文的结构有部分改动：
    1. more layers
    2. 所有的maxpooling替换成了stride2的separable conv
    3. 所有的depthwise conv后面都有BN+ReLU，和mobileNet里是一样的，DW-BN-ReLU-PW-BN-ReLU
       原始的SeparableConv2D则是[DW-PW]-BN-ReLU



## efficientNet B0-B7
    1. 从论文的acc pk上看，B0的精度就超过resnet了，B3及以上吊打resdeeper家族
    2. 但是我用B0实际实验（简单六分类）下来并没有
    3. B0-B7: finer & wider & deeper, input_size增大，channel加宽，layers加深
        |  network  | input_size |   width  |  deepth  | dropout_rate |
        |     B0    |     224    |    1.0   |    1.0   |      0.2     |
        |     B1    |     240    |    1.0   |    1.1   |      0.2     | 
        |     B2    |     260    |    1.1   |    1.2   |      0.3     | 
        |     B3    |     300    |    1.2   |    1.4   |      0.3     |
        |     B4    |     380    |    1.4   |    1.8   |      0.4     |
        |     B5    |     456    |    1.6   |    2.2   |      0.4     | 
        |     B6    |     528    |    1.8   |    2.6   |      0.5     |
        |     B7    |     600    |    2.0   |    3.1   |      0.5     |
    4. pip库: https://github.com/qubvel/efficientnet
        Requirements: 
        Keras >= 2.2.0 / TensorFlow >= 1.12.0
        keras_applications >= 1.0.7
        scikit-image

    5. todo：B4(https://www.kaggle.com/jimitshah777/bilinear-efficientnet-focal-loss-label-smoothing)


## efficientNet details
    1. input 
        input shape: >=32
        input value: imagenet_utils.preprocess_input输出[-1,1]

    2. scale
        round_filters: 基于width_coefficient加宽
        depth_divisor=8, 确保网络宽度是8的倍数
        round_repeats: 基于depth_coefficient加深

    3. DEFAULT_BLOCKS_ARGS
        kernel size 和 strides的值是针对depthwise conv的
        其余block都有固定的kernel size

    4. drop_connect_rate
        用在mbconv block里面，随着层数加深这个值逐渐增大
        dropout: 将输入单元的按比率随机设置为0，二维特征有一个noisy_shape参数
        rate: 是drop rate，不是keep prob
        noisy_shape: 
            是一个一维张量，和输入x的shape的长度一样，e.g. x_shape: [B,H,W,C] ---> noisy_shape: [?,?,?,?]
            其中的元素只能是1或者x的shape中对应元素，哪个轴为1，哪个轴就会被一致地dropout
            noise_shape=(None, 1, 1, 1)意味着我们希望一个batch的数据都遵循同一种的dropout模式————每个hwc的样本上被mute掉的元素位置相同

    5. residual condition
        if id_skip is True and strides==1 and filters_in==filters_out:

    6. dropout
        用在分类头上，

    7. activation
        全程swish
        swish最早出现在MobileNetV3，effNet用的block就是MobileNetV3的block，
        但是MobileNetV3不是只在深层block里面用了swish，因为论文里说实验下来这样好

    8. load_weights
        如果是加载ImageNet的预权重的话by_name=False，因为网络层的名字不一样


    9. convert weights
        h5解析: h5py._hl.dataset.Dataset数据结构和numpy数据结构








