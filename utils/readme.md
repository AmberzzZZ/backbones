## SoftSGD
    - 每隔一定的batch才更新一次参数，不更新梯度的step梯度不清空，执行累加，从而实现batchsize的变相扩大
    - 建议搭配间隔更新参数的BN层来使用，否则BN还是基于小batchsize来更新均值和方差


## SyncBN

