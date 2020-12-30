### custom formulation
    input: 256 / 192
    stem: 
        7x7 s2 conv
        residual
        maxpooling
        residual
        residual
    hourglass module
        4 levels features from x4 to x32, 3 times downsamp
        encoder: residual + maxpooling
        decoder: nearest upSamp + residual skip, residual the add
        mid: 3 residuals + residual skip
        head: 1x1 conv, n_channels output head & n_classes intermediate head
        next level input: residual, output head + intermediate head + current level input
        n_channels: keeping unchanged throughout
    intermediate supervision:
        n_classes intermediate head: heatmap for each category
        MSE loss


### formulation in cornerNet
    input: 512
    stem: 7x7 s2 conv ch128  &  7x7 s2 residual ch256
    encoder-decoder: 
        * ch[256, 384, 384, 384, 512]
        * residual: 1x1 ch/2, 3x3 ch/2, 1x1 ch
        * 5-times downsamp: residual + s2 conv
        * 5-time upsamp: 2 residual + nearest neighbor upsampling
        * skip connection: 2 residual, add
        * middle connection: 4 redidual
    intermediate supervision:
        * hourglass2 input：1x1 conv-BN to both input and output of hourglass1 + add + relu (intermediate supervision的结果不加回去)

    cornetNet hg back implementation: https://github.com/princeton-vl/CornerNet
    certerNet hg back implementation: https://github.com/xingyizhou/CenterNet