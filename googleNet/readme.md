## deeplab的xception相比原论文的结构有部分改动：
    1. more layers
    2. 所有的maxpooling替换成了stride2的separable conv
    3. 所有的depthwise conv后面都有BN+ReLU，和mobileNet里是一样的，DW-BN-ReLU-PW-BN-ReLU
       原始的SeparableConv2D则是[DW-PW]-BN-ReLU



## efficientNet B0-B7
    1. 从论文的acc pk上看，B0的精度就超过resnet了，B3及以上吊打resdeeper家族
    2. 但是我用B0实际实验（简单六分类）下来并没有
    3. B0-B7: finer & wider & deeper, input_size增大，channel加宽，layers加深
        |  network  | input_size |   width  |   depth  | dropout_rate |
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


## training details
    * RMSProp: decay=0.9, momentum(rho)=0.9
    * BN: momentum=0.99
    * weight decay = 1e-5
    * lr: initial=0.256, decays by 0.97 every 2.4 epochs
    * SiLU activation
    * AutoAugment
    * Stochastic depth: survive_prob = 0.8
    * dropout rate: 0.2 to 0.5 for B0 to B7


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
        dropout: 将输入单元的按比率随机设置为0，高维特征有一个noisy_shape参数
        rate: 是drop rate，不是keep prob
        noisy_shape: 
            是一个一维张量，和输入x的shape的长度一样，e.g. x_shape: [B,H,W,C] ---> noisy_shape: [?,?,?,?]
            其中的元素只能是1或者x的shape中对应元素，哪个轴为1，哪个轴就会被一致地dropout
            noise_shape=(None, 1, 1, 1)意味着我们希望一个batch内的数据dropout模式相互独立————每个hwc样本随机决定在当前这个residual path上通过/被截断置零

    5. residual condition
        if id_skip is True and strides==1 and filters_in==filters_out:

    6. dropout
        用在分类头上，还有backbone里面resnet block的id skip上
        每个stage都用，衰减
        skip上用dropout，residual上用se-block，都是为了强化语义信息
        why not dropblock？

    7. activation
        全程swish
        swish最早出现在MobileNetV3，effNet用的block就是MobileNetV3的block，
        但是MobileNetV3不是只在深层block里面用了swish，因为论文里说实验下来这样好

    8. load_weights
        如果是加载ImageNet的预权重的话by_name=False，因为网络层的名字不一样


    9. convert weights
        h5解析: h5py._hl.dataset.Dataset数据结构和numpy数据结构

    10. se-block
        no_bias: SENet原论文里面发现，去掉se-block的两个fc层的bias有助于特征学习
        但是eff官方权重里面是给se-block的conv带了bias的



## noisy-student & EfficientNet-L2

    top1 acc 88.4% with 300M additional unlabeled images

    teacher:
    * train with labeled images & standard CE at the begining
    * produce high-quality pseudo labels by reading in clean images
    * pseudo labels can be soft or hard
    * a larger and better teacher model leads to better results

    train student: 
    * larger efficientNet
    * not smaller than the teacher
    * optmize by CE on labeled & unlabeled images
    * common regularizations: dropout & stochastic depth & RandAugment

    pseudo data distribution:
    * select images that have confidence above 0.3
    * keep uniform distribution for each class
    * 样本多的类别，取highest confidence的
    * 样本少的类别，复制副本

    EfficientNet-L2
    * wider & deeper than EfficientNet-B7
    * but lower resolution
    |  network  | input_size |   width  |   depth  | dropout_rate |  test_reso  | 
    |     B7    |     600    |    2.0   |    3.1   |      0.5     |     600     |
    |     L2    |     475    |    4.3   |    5.3   |      0.5     |     800     |
    * finetune 1.5 epochs for test resolution on unaugmented labeled images

    training details
    * batch size: [512,1024,2048]都一样
    * 700 steps for model smaller than B4 & 350 steps for the largers
    * lr: 0.128 for batch 2048, decay by 0.97 every 2.4/4.8 epochs(for 350/700 steps)
    * stochastic prob: 0.8 for the last layer & linear decay rule
    * dropout: 0.5, for final layer
    * RandAugment: N=2, M=27



## efficientNetV2
    
    * 转权重的时候发现和论文结构不一致的地方：
        最后一个MB的输出channel是256，不是论文里的272
        top的第一个convBN的输出channel是1280，不是论文里的1792
    
    
    * settings
        RMSProp optimizer with decay 0.9 and momentum 0.9
        batch norm momentum 0.99
        weight decay 1e-5
        total epochs 350
        LR is first warmed up from 0 to 0.256, and then decayed by 0.97 every 2.4 epochs
        EMA with 0.9999 decay rate
        stochastic depth with 0.8 survival probability
        adaptive RandAugment, Mixup, Dropout: 4 stages (87 epoch per stage)
        inference: 
            maximum image size for training is about 20% smaller than inference & no further finetuning
            max inference 480 -> max train 394


    * uncertain: scaling policy
        compound scaling
        max inference image size 480
        gradually add more layers to later stages
        waiting for particular scale up factors...


    实际实验下来发现
    1. imageNet预权重贼重要，收敛/不收敛的区别
    2. progressive learning又慢又没用，直接训最后一个stage，用strong regularization，用effv1的训练策略就可以得到很好的结果










