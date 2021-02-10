## group conv
    keras和tensorflow里面没有group convolution（caffe和torch里面有）
    所以在构建静态图的时候，每层的C循环会影响效率，不知道实际计算资源分配的时候是不是并行的

    how keras implements the Group Conv:
    0. with feature input: [h,w,128], C=32, g_filters=4
    1. depthwise conv: depth_multiplier=g_filters, [h,w,128,4]
    2. reshape: splitting groups, [h,w,32,4,4]
    3. fuse: sum by groups, by axis-1, [h,w,32,4]
    4. reshape: flatten back to feature dims, [h,w,128]


## 写了一个keras的分组卷积层
    reference: https://github.com/fangwudi/Group-Depthwise-Conv3D-for-Keras
    参数量一样，但是构建图的时候比写循环快得多
    不知道对不对，需要实验对比一下
    tensorflow1.15, keras2.2.4
    //tensorflow2.0, keras2.3.1


## resnext
    cardinality: C               |   1  |   2  |   4  |   8  |   32  |
    width of bottleneck: kd      |  64  |  40  |  24  |  14  |    4  |
    width of group conv: filters |  64  |  80  |  96  | 112  |  128  |


## resneSt
    官方torch源码: https://github.com/zhanghang1989/ResNeSt
    第三方tf源码: https://github.com/QiaoranC/tf_ResNeSt_RegNet_model

    官方初始化：所有的卷积层权重normal init，所有的BN层权重one init，bias zero init

    官方下采样：avg pooling，resnet是stride2conv

    r=1, C=1: resnet
    r=1, C!=1: resnext
    r!=1: resneSt, r!=1才有split-attention模块，不然就是正常的3x3conv

    官方downsamp: 我看源代码将下采样放在第一层1x1卷积之后，本工程保持和resnet一致


## params
    depth50:
    r50     |   23,561,152
    se-r50  |   26,076,096
    rx50    |   23,082,240
    se-rx50 |   25,597,184
    rS50    |   15,894,192
    p3d-r50 |   27,482,240
    csp-rx50|   10,359,072
    sk-rx50 |   28,009,696
    gc-r50  |   26,110,096


## Stochastic Depth
    shrink the depth during training
    keep unchanged during testing

    traing phase randomly dropping resblocks: 
        只保留skip connection，残差通道mute掉
        Bernoulli: binary factor of active / inactive, survival probabilities, 服从uniform distribution / linear decay

    testing phase re-calibration: 
        跟dropout思路相似，model assemble的思路
        Hl = ReLU(prob*f(Hl-1,Wl) + Hl-1)


## pseudo 3d
    循环的abc block


## convert
    官方的h5文件，notop版本，3通道，有GlobalAveragePooling，conv层有bias
    dataset & group
    dataset结构和numpy差不多，用[:]可以转成array
    group结构类似dict，如果是biased conv，有kernel和bias两个权重参数，就需要通过key来读取


## cspresnet
    由于输入的通道减半，【计算量】减半，【参数量】减半，csp version可以不使用bottleneck结构
    leakyReLU
    origin skip: 1x1 conv s1/2, BN

    官方的cfg跟论文中说的不太一样，每个stage的block数和通道数都有变化，padding shortcut


## FuseConvBN层
    写了一个fuseconvbn，在inference阶段，bn中的mean和variance参数freeze掉了，
    相当于连续两个线性层，bn相当于1x1的卷积
    训练好以后可以构造一版fuse的模型，把权重转过去
    输出误差在小数点后6位


## SKNet
    based on resNext50
    M=2: number of kernel branches
    C=32: cardinality
    r=16: fc reduction ratio


## GCNet
    official implemention: https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
    based on ResNet50
    stage_with_gcb=(False, True, True, True)   # C3,C4,C5
    2 stage training:
        * 先训resnet50
        * 然后finetuning加了GC-block的resnet，frozen stem&stage1 layers












    