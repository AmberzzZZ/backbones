## 基本结构
    dense blocks: 每个block内部维持特征图尺寸不变
    transition blocks: 负责pooling降维

## dense blocks
    concat
    BN - ReLU - Kx3x3 conv
    bottleneck: 先用1x1 conv降维至固定4K，再3x3xK conv
    K = 32
    BN - ReLU - 4Kx1x1 conv - BN - ReLU - Kx3x3 conv
    用来学习特征的block

## transition layers
    BN - ReLU - 1x1 conv - 2x2 avg pooling
    compression: 1x1 conv降维至固定theta*K
    用来调整dims/resolution的block

## CSP-DenseNet
    partial split
    partial dense & partial transition






