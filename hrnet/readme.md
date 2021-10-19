## official repo
    
    * HRNet v1: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
    * HRNet v2: https://github.com/HRNet/HRNet-Semantic-Segmentation


## main innovations

    * parallel multi-resolution
    * repeated multi-scale fusions


## structure
    
    * stem: resnet stem
    * stage1: residuals
    * stage234: multi-resolution subnets

    * exchange unit
        * down path: 需要执行n次下采样，(n-1)个conv3x3-bn-relu + conv3x3-bn
        * id path: identity
        * up path: conv1x1-bn-upsample(nearest)


    * mask head
        所有的feature都一步上采样到highest resolution，然后concat

    * cls head
        每个level的feature逐层下采样，并与下一级feature相加


## models
    
    https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/0bbb288044/lib/config/hrnet_config.py




