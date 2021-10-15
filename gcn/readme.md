# layer level

## origin graph convolution layer

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
    
    就两层: 
    1. GCN-relu-dropout
    2. GCN-softmax


## GCN with trainable connectivity

    adjacent matrix也是个trainable variable



