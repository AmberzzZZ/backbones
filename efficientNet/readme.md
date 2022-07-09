
## efficientNet-lite

    main modifications:

    - fix stem & head: scaleup时候，这俩不动（不加宽），这里参数量减少
    - swish -> relu6: 限制浮点数分布，量化时候不损失精度
    - remove seblock: 移动端not well supported，这里参数量减少


