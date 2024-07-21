# 习作小模型

我想，没手搓过小模型并从原始数据进行过预训练就不能算是入门了这一代生成式AI吧。同样，要做一些实验，总还是需要一个试验田。

## 架构

* 模型：Llama
* 数据类型：
  * 主干： bfloat16
    * bfloat16精度很糟糕，在一些较宽的配置上不收敛或者收敛情况很糟糕
    * 需要配合梯度裁剪使用
  * softmax: float32
* Tokenizer：偷的ChatGLM3的tokenizer，词表大小为64K，我主要做中文，所以这个应该够用。

参数：

* 

## 训练配置

* 数据集：THUCNNEWS，清华大学出的一个早年的新浪网新闻的数据集，从种类上不是很平衡，但贵在语料质量高，不容易训崩。
* 