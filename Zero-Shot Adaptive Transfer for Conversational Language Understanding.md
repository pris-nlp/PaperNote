---
title: Zero-Shot  Adaptive  Transfer  for  Conversational  Language  Understanding
date: 2019-04-27 20:38:00
mathjax: true
tags:
- NLU
- Dialogue System
- Transfer Learning
categories:
- NLP
- NLU
thumbnail: http://imglf5.nosdn0.126.net/img/bG1jbzEvdHVjVjIxUDhYb2xFNHY2YnhEV1NVRjRoaTdoeFhwZ3VCSk1yS2NwVUUrSEF5Q1BnPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0
---
本文来自于AAAI2019，主要研究的是自然语言理解领域迁移的问题，与传统的序列标注模型不同的是，作者使用了slot描述信息来辅助多个领域之间的迁移，在10个领域的数据集上取得了最优的效果。

[paper link](https://drive.google.com/open?id=1DO8TnK4r2f3BxuSozf8ki0n1bemw0uKO)
<!-- more -->

## Introduction
序列标注任务是自然语言理解中的一个关键问题，智能对话代理（Alexa, Google Assistant, Cortana等）需要频繁地添加新领域识别的功能，而构建一个良好的序列标注模型需要大量的标注数据。因此，如何从已有的高资源领域迁移到低资源领域是一个很有意义的问题。

目前NLU迁移问题主要有两种方法：

* **data-driven**：将源数据集和目标数据集相结合，使用类似特征增强的方式进行多任务学习，参照[Fast and Scalable Expansion of Natural Language Understanding Functionality for Intelligent Agents](https://drive.google.com/open?id=1uThLZwYPgvGnsD_c0gosK6Ko0LJYc_dA)。缺点是数据集的增加会带来训练时间的增加。
* **model-driven**：与**data-driven**不同，**model-driven**并不直接利用源数据，而是将源模型的输出作为额外的特征添加到目标模型中，能够加快训练速度，参照 [Domain Attention with an Ensemble of Experts](https://drive.google.com/open?id=1SZoSEX79Z1Zi9Gml2Hqg3ao23mbPeuDb)。缺点在于**model-driven**需要显式地`concept alignments`即slot对齐。

本文提出的模型**Zero-Shot  Adaptive Transfer model  (ZAT)**借鉴于zero-shot learning，传统的序列标注任务把slot类型作为预测输出，而本文中的模型则是将slot描述信息作为模型输入，如下图：

![Figure  1:  (a)  Traditional  slot  tagging  approaches  with  the BIO  representation.  (b)  For  each  slot,  zero-shot  models  independently  detect  spans  that  contain  values  for  the  slot.  Detected  spans  are  then  merged  to  produce  a  final  prediction.](http://imglf6.nosdn0.126.net/img/bG1jbzEvdHVjVjJsZ0c3RWEvNDlzaTBmUlI4bGdRTnpLKy9oek5CYTFaenBRVnBsVStKZGZ3PT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

针对于同一个utterance，需要独立的经过每一类slot type模型预测结果，之后再把结果合并得到最终的输出。作者假设，不同的领域可以共享slot描述的语义信息，基于此，我们可以在大量的源数据中训练源模型，之后在少量的目标数据上finetune，并且不需要显式地slot对齐。

## Zero-Shot  Adaptive  Transfer  Model
![Figure  2:  Network  architecture  for  the  Zero-Shot  Adaptive  Transfer  model.](http://imglf5.nosdn0.126.net/img/bG1jbzEvdHVjVjIxUDhYb2xFNHY2YnhEV1NVRjRoaTdoeFhwZ3VCSk1yS2NwVUUrSEF5Q1BnPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

### Word  Embedding  Layer
对于input tokens和slot description tokens，ZAT模型使用了word embedding和character embedding拼接的方式，其中character embedding由CNN卷积然后max-pooling得到。

### Contextual  LSTM  Layer
得到token的编码之后，再经过一个Bi-LSTM编码层，注意input tokens和slot description tokens共享相同的Bi-LSTM层，分别得到隐层状态表示$X\in R^{d\times T}, Q\in R^{d\times J}$ 。

### Attention  Layer
注意力层的作用是获取input tokens的slot-aware的表征，使用每一个input token对应的隐层状态对slot description  tokens所有隐层状态做注意力：

![](http://imglf5.nosdn0.126.net/img/bG1jbzEvdHVjVjJsZ0c3RWEvNDlzbmI3a1Ezc1hReVdQOU8wZkE1Q2dEYTFrVnhoSS9nQVlnPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

其中$x_{t}$是$X$的第t行，$q_{j}$是$Q$的第j行。论文选择的$\alpha(x,q)=w
^{T}[x;q;x\circ q]$，最终得到$G_{:t}=\sum_{j}a_{tj}q_{j}$。

### Conditional  LSTM  Layer
然后逐元素求和计算$\mathbf{H}=\mathbf{G} \oplus \mathbf{X}$ ，再通过一个Bi-LSTM。

### Feedforward  Layer && CRF  Layer
最后通过一个前馈层和CRF层输出预测结果。

ZAT模型预测的时候需要将所有的slot description与input utterance经过模型，再将所有的结果合并。

> For  example,  we  merge  "Find $mexican_{category}$  deals  in  seattle"  and  “Find  mexican  deals in  $seattle_{location}$”  to  produce  “Find  $mexican_{category}$ deals  in  $seattle_{location}$.”  When  there  are  conflicting  spans, we  select  one  of  the  spans  at  random.

## Experiments
### Dataset

![Table  1:  List  of  domains  we  experimented  with.  80%  of  the  data  is  sampled  for  building  the  training  sets,  with  10%  each  for dev  and  test  sets.](http://imglf5.nosdn0.126.net/img/bG1jbzEvdHVjVjJsZ0c3RWEvNDlzdTVSQnF2aThncWVqSkd0My80eEZGdzI2cm0rOW1PQjV3PT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

### Domain  Adaptation  using  Zero-Shot  Model
针对于领域迁移任务，作者将10个领域数据集（随机取2000条）分为source data（包含9个领域）和target data（包含剩下的1个领域），先在source data训练出一个基础模型，之后再用target data进行finetue。
> Note that the size of the joint dataset for each target domain is 18,000, which is dramatically smaller than millions of examples used for training expert models in the BoE approach.

### Results  and  Discussion

![Table  2:  F1-scores  obtained  by  each  of  the  six  models  for  the  10  domains,  with  the  highest  score  in  each  row  marked  as  bold. Table  (a),  (b)  and  (c)  report  the  results  for  2000,  1000  and  500  training instances,  respectively.  The  average  improvement  is computed  over  the  CRF  model,  with  the  ones  marked * being  statistically  significant  with  p-value<0.05.](http://imglf6.nosdn0.126.net/img/bG1jbzEvdHVjVjJsZ0c3RWEvNDlzbkcwWlZZRGQ3WXVlZFVDc2kxMGx5R2hoVVFNY1k4UDJnPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

实验结果表明，ZAT模型与基线相比取得了最优的效果。

![Figure  7:  Visualization  of  attention  weights  for  the  input  sentence  ”Can  I  wear  jeans  to  a  casual  dinner?”  with  different  slots: (a)  category,  (b)  item,  and  (c)  time.](http://imglf3.nosdn0.126.net/img/bG1jbzEvdHVjVjJsZ0c3RWEvNDlzcGNwUGhvM0JTYTJhbFRONVMxdUFYSVNsNkxFeUY2eENRPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

## Conclusion
本文主要研究的是自然语言理解领域迁移的问题，提出了一种基于zero-shot learning的迁移方法，既避免了data-driven训练时间增加的缺点，同时也消除了slot对齐的问题，在各个领域的数据迁移实验中都取得了非常好的效果，尤其是低资源领域。
