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

    paper: GRAPH ATTENTION NETWORKS

    原始的GCN layer的图节点更新：
    * 左乘A，控制邻接节点名单，右乘W，计算邻接节点的加权和
    * A是一个fixed/trainable的矩阵变量
    * W是一个trainable的矩阵权重

    GAT的图节点更新：
    * 先用trainable weights W，将所有node feature投影到指定维度
    * 计算邻接节点的attention coefficients，softmax得到权重，再计算邻接节点的加权和
    * attention coefficients的计算是learnable的，通过一个FC+LeakyReLU的结构，hidden units=1
    * multi-head：搞K次上面的操作，concat起来作为结果，是single-head输出维度的K倍



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



