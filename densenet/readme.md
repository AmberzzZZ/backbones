## dense blocks
    concat
    BN - ReLU - 3x3 conv
    bottleneck: 先用1x1 conv降维至固定4K，再3x3xK conv
    K = 32

## transition layers
    BN - 1x1 conv - 2x2 avg pooling
    compression: 1x1 conv降维至固定theta*K
