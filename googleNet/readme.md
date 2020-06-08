## deeplab的xception相比原论文的结构有部分改动：
    1. more layers
    2. 所有的maxpooling替换成了stride2的separable conv
    3. 所有的depthwise conv后面都有BN+ReLU，和mobileNet里是一样的，DW-BN-ReLU-PW-BN-ReLU
       原始的SeparableConv2D则是[DW-PW]-BN-ReLU