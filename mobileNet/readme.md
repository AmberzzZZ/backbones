## 按照keras的源代码实现
    modelV1源代码与原论文的architecture有些不一致（stride的位置，block的个数）
    modelV2源论文中提到stride1的block有id path，stride2的block没有id path，没有提到输入输出通道数不对齐的情况
    源代码中实现是stride1且输入输出通道数一致的情况下有id path（能看到通道不对齐的的那几层基本都是stride2）
    modelV3源代码中se-block将最原始的se-block中的dense层改成1x1conv的升／降维，激活函数也有变化
    expansion上，原论文中给出的是exp size，源代码实现中用了ratio
    本代码与源代码的结构上简化了padding方式（源代码zeropadding+stride2conv+validpadding）
