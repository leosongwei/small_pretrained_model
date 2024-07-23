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

参数上分别考量了深和浅两种，效果比较微妙。看样子也许40层网络继续训练能超过16层网络？但我不打算投入更多机时了。

MobileLLM[1]论文鼓励了较深的层数，也许能有效？

![40vs16](./figures/output.png)

也训练了一个28层的，loss在两者之间，图都画出来比较乱就不放了。

参数对比：

```python
# 40层, hidden_size=512
# * Embedding与LM分类头各占参数量的7.9%
# * Decoder层占参数量的84%
self.config = custom_model.CustomModelConfig(
    vocab_size=self.tokenizer.vocab_size(),
    padding_token_id=self.tokenizer.token_pad_id,
    max_position_embeddings=4096,
    hidden_size=512,
    num_heads=16,
    MLP_intermediate=5000,
    num_layers=40,
    attention_dropout=0.1,
    dtype=torch.bfloat16,
    training=True,
    linear_imp = torch.nn.Linear
)
```

```python
# 16层, hidden_size=1024
# * Embedding与LM分类头各占参数量的14.8%
# * Decoder层占参数量的70%
self.config = custom_model.CustomModelConfig(
    vocab_size=self.tokenizer.vocab_size(),
    padding_token_id=self.tokenizer.token_pad_id,
    max_position_embeddings=4096,
    hidden_size=1024,
    num_heads=16,
    MLP_intermediate=5000,
    num_layers=16,
    attention_dropout=0.1,
    dtype=torch.bfloat16,
    training=True,
    linear_imp = torch.nn.Linear
)
```

## 训练配置

* 数据集：THUCNNEWS，清华大学出的一个早年的新浪网新闻的数据集，从种类上不是很平衡，但贵在语料质量高，不容易训崩。
* 优化器：bitsandbytes.optim.AdamW8bit
  * 感觉bitsandbytes这个AdamW里面加了一些魔法，更稳定的同时效果还更好
  * betas：(0.9, 0.95)
    * 考虑到默认值是(0.9, 0.999)，这算是相当激进的配置了。
* 学习率：1e-4
  * 试过1e-5，bf16基本训不动
  * 也试过更大的1e-3，和1e-4一样，很容易训炸。
  * MobileLLM提到了更大的学习率（2e-3）。
* 梯度裁剪的max norm设置为1.0
  * 否则1e-4的学习率也会训炸。
* batch size:
  * batch size = 16
  * 训练序列长度为192
  * gradient accumulation steps为20
  * 这样一次优化可以吃进61k tokens
  
## 效果

看起来能生成一点车轱辘话，还挺有意思的：

```txt
test input: 中兴手机
test out: ['中兴手机业务收入1000万 \n\u3000\u3000本报讯 (记者 田晓波) 昨日，中兴通讯(000002)发布2009年第三季度财报，公司实现营业收入1.15亿元，同比增长10.15%；归属于上市公司股东的净利润为1.01亿元，同比增长10.15%。\n\u3000\u3000中兴通讯2009年第三季度实现营业收入1.15亿元，同比增长1.1']
test input: 外交部网站22日发布
test out: ['外交部网站22日发布新闻\n\u3000\u3000新华网北京12月1日电 外交部发言人姜瑜1日说，外交部发言人马朝旭12日在例行记者会上表示，中方已就有关报道发表评论。\n\u3000\u3000有记者问：中方对此有何评论？\n\u3000\u3000有记者问：中方对此次访问有何评论？\n\u3000\u3000答：中方对此次访问有何评论？\n\u3000\u3000答：中方对此次访问，是第一次访问中国。\n\u3000\u3000问：中方']
test input: 基因组信息在生命科学研究中具有重要价值
test out: ['基因组信息在生命科学研究中具有重要价值\n\u3000\u3000新华网北京11月1日电(记者 魏?) 我国科学实验科学研究所2日宣布，我国将建立一个“科学实验”研究小组，并将在2012年10月1日，在“科学实验”中，将“科学”与“科学”结合，形成“科学”和“科学”的“科学”和“科学”等。\n\u3000\u3000“科学”是“科学”的“实验”\n\u3000']
```

## 引用

1. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases https://arxiv.org/abs/2402.14905