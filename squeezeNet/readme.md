### fire module
    squeeze: 
        1x1 conv, relu
        1x1 conv先降个维，save computation
    expand: 
        1x1 & 3x3 conv, relu
        用1x1和3x3两个分支提取特征
        最后concat


### 与inception module思路差不多，inception的人工定制痕迹更强一点


### by pass
    simple bypass: 
        identity skip, 
        fire3,5,7,9上添加(mid module, 输入输出通道数一样)
    complex bypass: 
        1x1conv skip, 
        fire3,5,7,9上添加simple by pass, 
        其余fire module上添加complex
    element-wise addition
