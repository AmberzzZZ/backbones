## resnet
    data gen: resize by [256,480] and crop by [224,224], subtract sper pixel meam
    structure: conv-bn-relu
    SGD: momentum=0.9
    lr: start from 0.1, divided by 10 when plateau
    weight decay: 1e-4, 用了较大的weight decay系数，比拥有strong regularization methods的efficientNet高了一个数量级
    do not use dropout!!! 原论文没用dropout


## resnet12 & resnet18
    活跃在few-shot learning领域的两个轻量back

    resnet18的结构图原论文里有:
    * baseblock不用bottleneck，而是两个3x3 conv
    * stage2-stage5的resblock重复两次

    few-shot paper TADAM's definition of ResNet-12:
    * It has 4 blocks of depth 3 with 3x3 kernels and shortcut connections. 
    * 2x2 max-pool is applied at the end of each block. 
    * Convolutional layer depth starts with 64 filters and is doubled after every max-pool
    * dropblock for stage3&4, dropblock_size=5
    

## group conv
    keras和tensorflow里面没有group convolution（caffe和torch里面有）
    所以在构建静态图的时候，每层的C循环会影响效率，不知道实际计算资源分配的时候是不是并行的
    参数量上用了group conv的conv层参数量会降低成原来的1/C

    how keras implements the Group Conv: (block3)[https://github.com/keras-team/keras/blob/0f8da5a7b814cb37baba868fc11fe8b10b3d4cf8/keras/applications/resnet.py]
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
    se154   |  115,671,592
    r50_rs  |   33,597,888


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


## SENet
    based on resNext
    在每个residual path的结尾添加se-block
    C=32: cardinality
    r=16: fc reduction ratio

    SOTA version: SENet-154
    * based on 64x4d ResNext-152
    * 每个bottleneck的第一个1x1 conv通道数减半: [g_filters//2, g_filters, g_filters]
    * stem是3个3x3conv
    * 1x1 s2conv替换成3x3 s2conv
    * dropout before final fc
    * wider: 64x4d说明g_filters start from 256
    * label smoothing
    * BN layers are frozen in the last few training steps
    [reference] https://github.com/qixuxiang/mmdetection_with_SENet154/blob/master/mmdet/models/backbones/senet.py


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


## ResNet_RS
    architecture:
    * stem的7x7conv换成3个3x3conv
    * stem的maxpooling去掉，每个stage的首个3x3conv负责stride2
    * residual path上前两个卷积的stride互换（在3x3上下采样）
    * id path上的1x1 s2conv替换成2x2 s2的avg pooling+1x1conv
    * use se-ratio of 0.25

    参数量比se-r50多: 主要因为se_ratio变了

    hyper:
    * label smoothing = 0.1
    * weight decay = 4e-5
    * dropout rate: 标准network width下是0.25，wider increase，vice versa
    * stochastic drop rate: 0.2 for resolutions 224 and above

    training settings from paper:
    * regularization for 10 and 100 epochs: flips & crops, weight decay
    * regularization longer training epochs: RandAugment, Dropout, Stochastic Depth, Label Smoothing, weight decay
    * lr schedule: cosine
    * optimizer: SGD-M, 
    * EMA = 0.9999, both for weights and BN

    training settings from source code:
    * BN momentum=0.9
    * weight_decay implementation: loss = cross_entropy + weight_decay*sum([l2_loss(i) for i in trainable_params_exclude_bn])
    * EMA=0.9999 implementation: ema on the the trainable variables, include bn
    * SGDM momentum=0.9
    * scheduled drop_prob: 随着training step变化
    * 先做一次梯度更新，然后对新的权重计算EMA，用EMA的结果作为最终的权重













    