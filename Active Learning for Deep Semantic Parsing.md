## 《Active Learning for Deep Semantic Parsing》阅读笔记

### 要解决的问题
语义分析任务（Semantic Parsing）指的是：从给定的自然语言查询（query）中提取有效信息，将其转换为一个逻辑范式（Logical Form, LF）。也就是需要将表述方式十分随意的句子转换成一个结构化的、计算机能理解的形式语言句子。需要转换成的LF的形式取决于具体任务。下图是对应到某些任务（数据集）中的**自然语言 - LF**的样例：
![图片](http://imglf5.nosdn0.126.net/img/Y3cva2JCQ0tUWi91TkhqKzN0OUdkQWFONkhOSHZmaS9JUXNtTFBnQklEZXdxcGFLcEtkeU1RPT0.jpg)
可以看到被转换的LF也是一个句子（序列），因此其本质上也是一个seq2seq的任务。

这类任务往往需要花费巨大的代价对数据进行标注，因此本文旨在通过主动学习减少所需的人工标注数据量。

### 标注数据的两种方式
对于语义分析这一任务而言，目前主要有两种获取标注数据的方式：
- 传统的方式首先获取自然语言句子，然后通过定义好的LF形式进行标注
- 另外一种方式，来自于一篇论文《Building a Semantic Parser Overnight》，因此称为overnight标注方法。它相当于是倒过来，先通过定义好的文法生成所有的LF，然后让标注人员将这些LF表述成自然语言查询，如下图所示：
  ![图片](http://imglf4.nosdn0.126.net/img/Y3cva2JCQ0tUWi91TkhqKzN0OUdkSW9YRzJQbE1SWVJnUG1SZW42T3c1N1Y3VmkrblB1dW1RPT0.jpg)

前一种传统的数据标记方式能够方便地应用到传统主动学习策略（例如least confidence、large margin、entropy based等等方式）去做sample selection。而后一种数据标记方式（overnight）由于自然语言样本x是未知的（是需要使用人力去生成的），因此就无法直接基于P(y\|x)引用least confidence策略去做主动学习，因此本文主要聚焦在对于第二种标记方式（overnight）的主动学习策略的探索上。

### 基础模型
本文做语义分析任务的模型就是传统的seq2seq模型，使用一个双向RNN编码序列，再用一个双向RNN解码序列。训练时需要最小化的loss为负对数损失：
![图片](http://imglf6.nosdn0.126.net/img/Y3cva2JCQ0tUWi91TkhqKzN0OUdkSHlYOC9iZ25qTFI1MHpxWUhYUTFZT0hWMHVaMDUzQ2pnPT0.jpg?)

### 主动学习方法

#### 针对传统数据集的主动学习策略
对于传统数据集，一个简单但有效的主动学习策略就是基于最小置信度，它是从未标注数据集中选取模型预测序列不确定性最高的样本进行标注：
![图片](http://imglf6.nosdn0.126.net/img/Y3cva2JCQ0tUWi91TkhqKzN0OUdkTDd2SnVxSFh6WnNhOVpDTHFCWTEvdVdac1RndEdsL1lBPT0.jpg)
对应到上述基础模型，相当于就是从未标注数据集中选取预测的负对数损失(1)最大的样本进行标注。

考虑到通过最小置信度进行sample selection使用的信息过于单一，作者还尝试了训练一个分类器去预测模型生成目标LF序列错误的概率，最后选取错误概率最高的样本进行标注即可。作者使用了许多不同的特征作为分类器的输入，包括对数损失、最优和次优预测的margin、句子频度（这个没懂，原文是source sentence frequency）、编码器的最终状态、解码器的最终状态等；同时还使用了不同的分类器结构，包括逻辑回归、全连接神经网络、多层CNN等等，在已标注的数据集上进行训练，但最后（在dev数据上）发现最小置信度策略的效果和分类器是不相上下的。也就是说使用最小置信度策略，在传统的数据集上就够用了。

#### 针对overnight数据集的主动学习策略
由于overnight的数据标注方式是基于给定的LF(y)去产生自然语言句子(x)，因此主动学习的任务就变成了：如何从未标注的LF集合中选取LF来获取自然语言句子，能够最快提升模型性能。

上面提到，由于overnight这种标注方式的特殊之处，基于P(y\|x)应用least confidence的策略是不可行的，所以需要探索别的主动学习策略。

其中一种直观的方法是使用反向的S2S对P(x\|y)进行建模，即给定LF(y)，预测自然语言(x)。这样就能基于P(x\|y)应用least confidence了，如下图：
![图片](http://imglf3.nosdn0.126.net/img/Y3cva2JCQ0tUWi91TkhqKzN0OUdkSHlYOC9iZ25qTFJQY0orSTdaQ3RLckkzRkpRb0M5N2V3PT0.jpg?)
但这种方法忽略了一个重要的问题，就是LF其实是自然语言的抽象表示，一个自然语言句子只对应了一个LF，但LF可能对应多个不同的自然语言句子（同一语义的多种表述），从而导致了其置信度比较低（因为有多种自然语言表述都有可能）。因此这种基于P(x\|y)应用least confidence的方法是有缺陷的。

因此作者提出了另外一个方法：训练一个二元分类器去预测样本是否已经被选入已标记数据集中。作者使用了两组特征作为分类器的输入：
1. LF model: 是使用LF句子训练的一个在LF上的语言模型，去对P(y)进行建模，该模型最后输出的log(P(y))将会作为分类器的特征
2. backword S2S model: 就是使用seq2seq模型去对P(x\|y)进行建模，即给定LF，要求预测自然语言句子，使用的特征和上面提到的forward S2S model的类似。

在针对dev数据的测试中，作者发现使用source LF frequencies和margin of best and second best solution这两种特征的融合，训练出来的分类器效果是最好的。因此最后就使用这两个特征的线性组合作为分类器。

PS: 这篇论文作者通篇都没有放模型图，完全就是通过语言对模型的描述...

### 实验
![图片](http://imglf5.nosdn0.126.net/img/Y3cva2JCQ0tUWi91TkhqKzN0OUdkSHlYOC9iZ25qTFI3NEJZUHNGVEc5ZldJWjdpMXRJa09nPT0.jpg?)

作者在三个semantic parsing的数据集进行实验，其中(c)就是overnight的数据集。对比试验分别采用了：1) Random - 随机selection；2) Fw S2S - 使用seq2seq模型建模P(y\|x)；3) Bw S2S - 使用seq2seq模型建模P(x\|y)，使用其log loss做selection；4) Bw classifier - 针对overnight数据集，上面提到的多种特征的线性组合来训练分类器。采用的评判标准是曲线下面积。

可以看到在图(c)中，Bw classifier的效果和Fw S2S很接近，证明此方法是有一定效果的。

### 待思考的点
- 本文提出的第二种判别器的方法和之前讲的那种基于对抗的主动学习方法是同一种吗？若不是，它们有何不同呢？
- 前向和反向seq2seq模型，分别建模了P(y\|x)和P(x\|y)，从概率的意义上来讲有什么不同？这两者模型之间是否存在某种对应关系or某种程度上的等价？
- 在使用前向或反向seq2seq建模P(y\|x)或者P(x\|y)时，从模型的不同结点处提取出来多个特征有什么意义吗？这些特征之间包含多大程度的独立信息和冗余呢？