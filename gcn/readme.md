# layer level

## origin graph convolution layer

    paper: SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS

    layer inputs: 
        batch inputs: (b,N,D)
        adj matrix: (b,N,N), repeat along b-axis

    layer weights:
        weights: (D,F)
        bias: (F,)

    layer outputs:
        batch outputs: (b,N,F)

    call: 
        左乘adj mat，右乘weights mat


## graph convolution layer with residual path

    paper: DAG: Structured Landmark Detection via Topology-Adapting Deep Graph Learning

    layer inputs: 
        batch inputs: (b,N,D)
        adj matrix: (b,N,N), repeat along b-axis

    layer weights:
        weights for forward path: (D,F)
        bias for forward path: (F,)
        weights for residual path: (D,F)
        bias for residual path: (F,)

    layer outputs:
        batch outputs: (b,N,F)

    call:
        residual path: fc（右乘weights_res）
        forward path: 左乘adj mat，右乘weights mat
        add


## graph attention




# network level

## orig paper's GCN
    
    task: classification

    就两层: 
    1. GCN-relu-dropout
    2. GCN-softmax

    vis: TSNE


## GCN with trainable connectivity

    task: coordinate regression based landmark detection

    多个GCN cascade逼近精确预测坐标
    * inputs：visual features (b,h,w,c) 和 initial graph (n,n)
    * 一个GCN-global：预测一个透视变换参数，(b,9)
    * 多个GCN-local：预测一个坐标偏移量，(b,N,2)

    adjacent matrix也是个trainable variable



