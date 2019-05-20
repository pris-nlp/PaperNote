## 《Cross-Lingual Transfer Learning for Task Oriented Dialog》阅读笔记

Conference from: NAACL2019

Paper link: [https://arxiv.org/pdf/1810.13327](https://arxiv.org/pdf/1810.13327)

### 简介

本文主要想解决的问题是SLU（Spoken Language Understanding）的跨语言迁移，即从高资源语言SLU迁移到低资源语言SLU。本文的贡献有：
- 收集了新的一个跨语言（英语、西班牙语和泰语，其中英语为源语言，西班牙语和泰语为目标低资源语言）、多任务（三种任务：Weather、Alarm、Reminder）的数据集
- 提出了一种基于预训练编码器的跨语言迁移方法
- 基于该数据集，在传统的两个方法（翻译训练数据、使用跨语言词向量编码）和本文提出的基于预训练编码器的迁移方法上进行了实验

### 提出的数据集

数据集发布在[https://fb.me/multilingual_task_oriented_data](https://fb.me/multilingual_task_oriented_data)。该数据集的统计数据如下图所示：
![数据集统计数据](https://i.loli.net/2019/05/19/5ce117bfac6c396045.png)

数据集的收集过程为：首先让英语母语者为每一个intent生产句子（比如：会怎么样询问天气），这样收集了43000条英语句子；随后让两个标注者基于此标注intent和slots，如果两者有分歧，则请求第三个标注着来做最终裁定；对于西班牙语和泰语的样本，则是让一小部分母语者将英语的一部分样本（随机采样）翻译成对应的语言；对于西班牙语的标注同上，出现分歧时就让第三个同时精通于英语和西班牙语的标注着进行裁定；而对于泰语，由于找不到同时精通于英语和泰语的标注者，所以处理方式就是直接丢弃掉出现标注分歧的样本。

### NLU模型

#### 基础NLU模型

本文的基础NLU模型主要包含两个部分：
1. 首先会让给定句子通过一个句子分类器，去识别该句子属于哪个领域（Alarm、Reminder、Weather之一）
2. 让这一个特定领域的模型去联合预测intent和slots

![基础NLU模型图](https://i.loli.net/2019/05/19/5ce1340da71d357848.png)
上图展示了基础模型的结构：首先每个单词会被编码成一个向量，随后每个时间步的向量通过一层双向LSTM。对于意图识别而言，所有时间步的隐层向量会经过一个self-attention，然后被丢入softmax层去预测这句话的意图；对于槽填充，会将每个时间步的隐层向量经过softmax层后丢入CRF层进行预测。

而实验主要对照的点在于**词向量编码层**，可选方案如下：
- **Zero embeddings**: 即使用一个词向量矩阵随任务进行训练，这个矩阵将会在一开始被初始化为零
- **XLU embeddings**: 使用预训练好的一个跨语言的词向量矩阵（称为XLU embeddings，见引文：[Ruder et al., 2017, A survey of cross-lingual word embedding models](https://arxiv.org/pdf/1706.04902)）编码一个词，并与上述随任务训练的zero embeddings进行拼接，这里的跨语言词向量矩阵是fixed的
- **Encoder embeddings**: 使用一个通过某种方法（后文会介绍）预训练的双向LSTM句子编码器，提取其最上层的隐层向量作为该句子中每个词语的表示，并且将这些向量与随任务联合训练的zero embeddings拼接，作为这个词最后的表示

#### 预训练的编码器模型

上文提到的encoder embeddings需要用到一个预训练的双向LSTM句子编码器。在本文的所有实验中，均采用了两层的双向LSTM编码器。实验比较了以下三种具体的策略（模型结构和预训练目标）：
- **CoVe**: 按照引文[McCann et al, 2017, Learned in translation: Contextualized word vectors](https://arxiv.org/abs/1708.00107)，训练一个机器翻译模型，将低资源的语言（西班牙语或泰语）翻译成英语，然后将机器翻译模型中的编码器作为句子编码器
- **Multilingual CoVe**: 训练一个机器翻译模型，能够同时将低资源的语言翻译成英语和把英语翻译成低资源语言，模型的翻译方向取决于解码器的第一个和目标语言相关的输入token（详细见引文[Yu et al., 2018a, Multilingual seq2seq training with similarity loss for cross-lingual document classification](http://www.aclweb.org/anthology/W18-3023)）。在预训练这一模型的过程中，编码器是语言不可知的（即编码器无法获知所翻译的句子具体属于何种语言），因此可以期望模型学到跨语言的语义特征
- **Multilingual CoVe w/ autoencoder**: 作者使用的是一个双向的机器翻译模型，同时联合了自编解码器的训练目标。比如对西班牙语-英语的句子对而言：给定西班牙语的输入句子，模型会根据解码器输入的第一个token，要么生成对应的英语翻译，要么生成这个句子本身。给定英语输入句子也同样，解码器应该根据第一个token，要么输出其对应的西班牙语翻译，要么重现这个句子本身。这样设计训练目标的动机是：让编码器学习到泛化能力更强的跨语言的语义表示，因为这里和上一种训练方式不同，输入句子的语言并不决定输出句子的语言

此外，对于西班牙语，作者还使用了预训练的ELMo编码作为对照，但西班牙语的ELMo编码相当于仅仅是在西班牙语的语料上进行预训练的，所以它并不是跨语言的编码。

#### 预训练模型的结果

下图给出了各个预训练模型的perplexity指标：
![预训练模型的perplexity指标](https://i.loli.net/2019/05/19/5ce1427c2f50164490.png)

其中perplexity指标是用来评价基于概率的生成模型，其生成样本好坏程度的一个方法，见[维基百科](https://en.wikipedia.org/wiki/Perplexity)。

### 实验及结果分析

#### 实验一：跨语言学习的不同迁移方法

作者首先在以下设定下使用基础模型进行了实验：
- **Target only**: 只使用低资源的目标语言作为训练样本
- **Target only with encoder embeddings**: 只使用低资源的目标语言作为训练样本，但是其编码层采用预训练的encoder embeddings
- **Translate train**: 将英语的训练样本翻译到目标语言，并与目标语言的训练样本融合，其中机器翻译采用的是Facebook的机器翻译系统，标注的slot信息通过attention权重（引文：[Yarowsky et al., 2001, Inducing multilingual text analysis tools via robust projection across aligned corpora](https://www.aclweb.org/anthology/H01-1035)）映射到翻译后的句子
- **Cross-lingual with XLU embeddings**: 将英语和目标语言的训练样本混合后进行训练，并采用XLU embeddings编码token，其中XLU embeddings采用的是预训练的MUSE跨语言编码（引文：[Conneau et al., 2017, Word translation without parallel data](https://arxiv.org/abs/1710.04087)），由于该编码没有对泰语的版本，所以只在西班牙语上进行了实验
- **Cross-lingual with encoder embeddings**: 将英语和目标语言的训练样本混合后进行训练，并采用上述的三种encoder embeddings编码token，同时作为对照的ELMo编码也会在西班牙语上进行实验

实验结果如下图所示：
![第一个实验的实验结果](https://i.loli.net/2019/05/19/5ce150242456e17293.png)
可以看到两种语言的实验结果略有差别，但也有一些共同的表现。

对西班牙语而言，使用target only的训练数据，语境化的词语表示都获得了更好的性能，其中ELMo由于在庞大的单语言语料上进行预训练，取得了最好的效果；而如果看跨语言的训练表现，会发现翻译训练语料的方法在意图分类和领域分类上取得了更好的效果，但对于槽填充则没有，作者认为原因可能是在slot映射的时候引入了噪声。

总体上来看，使用跨语言的训练方法都相比target only的设定取得了更好的性能，但采用何种embedding type，对模型的最终性能相对来说影响不大。

#### 实验二：Zero-shot的学习，及其学习曲线

从上一个实验看来，单单使用全部的目标语言训练样本，无法清楚的看出跨语言的编码是否有帮助。因此作者做了第二个实验，也就是使用更少的数据量的学习。

首先是zero-shot的实验结果，即：不使用任何目标语言训练样本，单单采用英语的训练数据，其实验结果如下图：
![Zero-shot实验结果](https://i.loli.net/2019/05/19/5ce15657f253e62868.png)
可以看到预训练的两种跨语言CoVe编码是超过单语言的CoVe编码的，证明了跨语言的编码相比单语言编码还是有所帮助的，同时可以在西班牙语的实验结果上看到这两种跨语言的CoVe编码，在zero-shot的情况下比XLU embeddings的效果更好。另外，意料之中，机器翻译（Translate train）的方法在zero-shot时取得了最好的效果。

其次是在有限的目标语言（few-shot）训练数据下，模型的性能表现如下：
![不同数量的目标语言训练样本下的实验结果](https://i.loli.net/2019/05/19/5ce15797f107f25400.png)
实验结果验证了跨语言的编码在few-shot的情形下的确是有帮助的（与两种Target only的编码方法相对比）。
