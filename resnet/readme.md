## group conv
    keras和tensorflow里面没有group convolution（caffe和torch里面有）
    所以在构建静态图的时候，每层的C循环会影响效率，不知道实际计算资源分配的时候是不是并行的


## 写了一个keras的分组卷积层
    reference: https://github.com/fangwudi/Group-Depthwise-Conv3D-for-Keras
    参数量一样，但是构建图的时候比写循环快得多
    不知道对不对，需要实验对比一下


## resnest
    官方torch源码: https://github.com/zhanghang1989/ResNeSt
    第三方tf源码: https://github.com/QiaoranC/tf_ResNeSt_RegNet_model


## Stochastic Depth

    