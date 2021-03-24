## SoftSGD
    - 每隔一定的batch才更新一次参数，不更新梯度的step梯度不清空，执行累加，从而实现batchsize的变相扩大
    - 建议搭配间隔更新参数的BN层来使用，否则BN还是基于小batchsize来更新均值和方差


## SoftBN
    - inspired by SoftSGD & syncBN
    - 多卡同步的时候，计算几个mini-batch的均值和方差，来affine每个不同的sample-batch
    - 在串行场景下，同时在SoftSGD设置下，每间隔n个step才更新一次梯度，也就是说在间隔内的step中，神经元在做一样的线性运算，理论上是可以合并的————renorm
    - 考虑一个short-term mean&var
    - 每隔一定的step才更新一次参数，不对外更新mean&var的step，在内部执行mean&var的累加
    - 相当于一个串行的分布式BN


## SyncBN
    - 用keras的multi_gpu方法进行单机多卡训练，多卡模型上的BN本身就是同步的，BN layer的四个参数在构建图的时候被声明，在每一个step每个模型的所有参数都保持一致
    - 分布式训练的情况下，需要显式地自行配置


## AdamW
    Adam with weight decay
    将weight decay集成进keras的optimizer，替换layer regularization

