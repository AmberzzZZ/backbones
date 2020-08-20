## vs AlexNet
    主要区别就是加深了，用连续的小尺寸卷积核代替大尺寸

## vgg结构
    结构非常统一，3x3的卷积+2x2的pooling+全连接
    vgg16: 图中D
    vgg19: 图中E

    LRN: 不多说了，“从后面的ResNet、DenseNet、SENet等网络就能看出LRN的影响力并不大，已经被drop、BN所取代”

    可添加: BN，Dropout

## 缺点
    三个全连接层导致参数量贼大，慢的一匹
    vgg16: 134,301,514
    vgg19: 139,611,210
    85%以上的参数量都来自于全连接层

## SSD back
    就是看SSD里面用了它，才记录一下，SSD用vgg16做backbone，
    * 前四层conv+pooling和vgg16一样，conv4输出38x38x512
    * 第五个conv之后的pooling由2x2 s2变成 3x3 s1，输出19x19x1024
    * 接下来的尾巴是新增的
        ** conv6是3x3x1024的空洞卷积，输出19x19x1024
        ** conv7是1x1x1024的conv，输出19x19x1024
        ** conv8是1x1x256和3x3x512 s2的conv，输出10x10x512
        ** conv9都是1x1x128和3x3x256 s2的conv，输出5x5x256
        ** conv10、conv11都是1x1x128和3x3x256 s1 p0的conv，输出3x3x256和1x1x256
    多尺度输出: conv4 conv7 conv8 conv9 conv10 conv11都接上检测头作为输出

