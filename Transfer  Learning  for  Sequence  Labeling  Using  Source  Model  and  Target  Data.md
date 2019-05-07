---
title: Transfer  Learning  for  Sequence  Labeling  Using  Source  Model  and  Target  Data
date: 2019-04-12 19:48:00
mathjax: true
tags:
- Transfer  Learning
- Sequence  Labeling
categories:
- NLP
thumbnail: http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjJ5NnVkangrMnRtVE0vSTdHTUw0eGNGQWxjM045K3p1UTh0aVpCaWVWcGtRPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0
---
本文来自于AAAI2019，主要研究的是迁移学习在序列标注任务上的应用，仅仅基于source data训练的source model迁移到新的target data（与source data相比，增加了标签的类别，而领域不变），而不直接使用source data来迁移，实验结果证明迁移学习在新标签类别和之前已有的标签类别上都取得了不错的效果。

[paper link](https://drive.google.com/open?id=1mEDVgr_ZWr58GCFA6ZJu-3uAJVZ1oHpF)
[code link](https://github.com/liah-chan/transferNER)
<!-- more -->

## Introduction
实际的序列标注任务往往会存在实体类别动态改变的问题，以NER为例，金融领域可能会存在*Companies  or  Banks*实体，政治领域存在*Senators, Bills,  Ministries*类似实体，除了这些领域特定的实体类别，还存在一些通用实体，例如*Location  or  Date*。因此，基于通用实体的标注数据来迁移到特定领域，增加一些领域特定的实体类别，是一个很有意义的研究问题。并且，针对于一个固定的领域，领域的实体类别也是有可能变化的，例如新产品介绍。显然，针对新出现的实体类别进行大量的数据标注和重训练是不可行的。

为了简化模型，作者做了以下设定：

* 基于source data $D_{s}$ 训练出source model $M_{s}$
* 定义一个迁移学习任务TL：从source data $D_{s}$ 迁移到target data $D_{t}$ ，注意$D_{t}$中除了$D_{s}$已有的实体类别之外，新增了一些实体类别，但是$D_{t}$的规模远远小于$D_{s}$，并且迁移的时候不允许直接使用$D_{s}$训练$M_{t}$。

本文提出了一种渐进式的序列标注模型，以解决上述问题。模型主要分为两部分：

* 给定在source data $D_{s}$ 训练出的source model $M_{s}$（实际使用的是Bi-LSTM+CRF），使用其参数来初始化$M_{t}$，同时增加$M_{t}$输出层的维度，然后在target data $D_{t}$ 上fine-tuning。
* 增加了一个neural adapter来连接$M_{s}$和$M_{t}$，通过一个Bi-LSTM来实现，以$M_{s}$的最后线性层输出（未经过softmax）为Bi-LSTM的输入，它的输出作为$M_{t}$的额外输入。适配器adapter的主要作用是解决$D_{s}$和$D_{t}$中标签序列不一致的问题。
	> the  surface  form  of  a  new  category  type  has already  appeared  in  the  $D_{S}$,  but  they  are  not  annotated  as a  label.  Because  it  is  not  yet  considered  as  a  concept  to  be recognized.

以上过程在训练$M_{t}$时，$M_{s}$的参数都是固定的。

## Progressive  Adaptation  Models

### State-of-the-art  in  Neural  Sequence  Labeling

![Figure  1:  Source  and  target  model  architecture](http://imglf3.nosdn0.126.net/img/bG1jbzEvdHVjVjJiYmVDbGNtaVpSbWZRTjE3MWd3SEFwRUdTZlRoR3NXS3NyZWk4aE5QbXBnPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

作者在source model和target model使用了相同的网络结构Bi-LSTM+CRF，只是最后的输出层维度增加，其余相同。

###  Problem  Formalization

> In  the  initial  phase,  a  sequence  labeling  model,  $M_{S}$,  is trained  on  a  source  dataset,  $D_{S}$,  which  has  E  classes.  Then, in  the  next  phase,  a  new  model,  $M_{T}$,  needs  to  be  learned  on target  dataset,  $D_{T}$,  which  contains  new  input  examples  and E  +  M  classes,  where  M  is  the  number  of  new  classes.  $D_{S}$ cannot  be  used  for  training  $M_{T}$.

### Transfer  Learning  Approach

**Training  of  a  source  model**:

![](http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjJiYmVDbGNtaVpSbDh5ZlJIaEZ5R21OVnJyakpobUNHMWdzcTVxYk5oQ09RPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

**Parameter  Transfer**: 因为增加了新的类别，所以要修改Bi-LSTM后的最后一层FC的维度，如Figure 1所示。具体来说，FC的作用是将LSTM的输出隐层向量**h**映射到维度为 $nE+1$ 的向量**p**，其中n是由标注格式确定的一个常数因子，对于BIO格式（*B-NE*,*I-NE*）来说$n=2$，而增加了M个新类别后，FC的输出维度应该增加为 $n(E+M)+1$。对于要修改维度的FC层，其参数初始化由$X \sim \mathcal{N}\left(\mu, \sigma^{2}\right)$ 得到，其中$\mu, \sigma$ 是原FC权重参数的均值和标准差；而对于其它尺寸没有变化的网络层，直接用$M_{T}$对应的层初始化，如下所示。

![](http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjJiYmVDbGNtaVpSc1N6Z2VXekxsaUJKb25ZQWJ6enFaUzhHWk0vQ3FDTEFBPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

**Training  the  target  model**: 

![](http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjJiYmVDbGNtaVpScUp1L0ZONmYraXlibWxHcFJ4K1htSXNmaVJ1UmhxcTdnPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

### Transfer  Learning  using  neural  adapters

> It should be noted that many word sequences corresponding to new NE categories can already appear in the source data, but they are annotated as null since their label is not part of the source data annotation yet. This  is  a  critical  aspect  to solve  as  otherwise  the  target  model  with  transferred  parameters  would  treat  the  word  sequence  corresponding  to  a  new NE  category  as  a  null  category.

![Figure  2:  Our  Proposed  Neural  Adapter](http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjJ5NnVkangrMnRtVE0vSTdHTUw0eGNGQWxjM045K3p1UTh0aVpCaWVWcGtRPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

适配器adapter的主要作用是解决$D_{s}$和$D_{t}$中标签序列不一致的问题。以$M_{s}$的最后线性层输出（未经过softmax）为Bi-LSTM的输入，Bi-LSTM的输出作为$M_{t}$的额外输入。

$$\overrightarrow{a}_{t}=\overrightarrow{\mathrm{A}}\left(p_{t}^{\mathrm{S}}, \overrightarrow{a}_{t-1}\right)$$

$$\overleftarrow{a}_{t}=\overleftarrow{\mathrm{A}}\left(p_{t}^{\mathrm{S}}, \overleftarrow{a}_{t+1}\right)$$

$$\boldsymbol{p}_{t}^{\mathrm{T}^{\prime}}=\boldsymbol{a}_{t} \oplus \boldsymbol{p}_{t}^{\mathrm{T}}$$

$$\boldsymbol{a}_{t}=\left[\overrightarrow{a}_{t} \oplus \overleftarrow{a}_{t}\right], \oplus \text{ is the element-wise  summation}$$

得到$\boldsymbol{p}_{t}^{\mathrm{T}^{\prime}}$后，再经过softmax归一化得到输出概率分布。整个过程中$M_{S}$的参数是固定不变的。

> The choice of BLSTM as the adapter is motivated by the fact that we want to incorporate the context information of a feature in the sequence to detect the new category that was annotated and possibly incorrectly predicted as not a label.

## Experiments

### Datasets
作者使用了[CONLL 2003 NER](https://www.clips.uantwerpen.be/conll2003/ner/)数据集，原始的数据集包括四类实体：organization(ORG),  person(PER),  location(LOC)  and  miscellaneous(MISC) 。针对于本文提出的任务，作者按照8/2划分$D_{S}$和$D_{T}$，然后针对于每一类实体，分别在$D_{S}$中标注成O，而在$D_{T}$中保留，这样可以得到四个数据集。同时，为了验证本文提出的模型在不同语言上的效果，作者还使用了[I-CAB  (Italian  Content  Annotation  Bank)](http://ontotext.fbk.eu/icab.html)。

![Table  1:  Number  of  entities  in  CONLL  dataset  (in  English)  and I-CAB  dataset  (in  Italian).](http://imglf6.nosdn0.126.net/img/bG1jbzEvdHVjVjNadWN1K1dJMnRkZzRGN3I3QUZ1b2pXbENPbzJQU05oQkUvc3JqS3ZTVk5BPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

### Results  on  CoNLL  and  I-CAB  datasets

![Table  2:  Performance  of  the  source  model  (MS)  and  the  target  model  (MT),  according  to  different  settings.  The  reported  performance  is the  F1  score  on  the  test  set.  Ori.  indicates  the  original  3  NE  categories  in  the  source  data,  while  New  indicates  the  new  NE  categories  in  the target  data.  All  is  the  overall  test  F1  in  the  subsequent  step  (for  all  4  NE  categories).](http://imglf5.nosdn0.126.net/img/bG1jbzEvdHVjVjNadWN1K1dJMnRkcDdjT2ZzQU9sbndXSmFCKzI4UnhWK1NXSG51T1J3SGVnPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

* 参数迁移可以带来更好的结果
* 固定参数会导致结果变差，尤其是新的实体类别
* 使用adapter会带来效果提升

![](http://imglf3.nosdn0.126.net/img/bG1jbzEvdHVjVjNadWN1K1dJMnRkbUtacUVIRWkvb3JwanVqTVlLWnpnNnp1eUhxUUNCRERBPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

![](http://imglf5.nosdn0.126.net/img/bG1jbzEvdHVjVjNadWN1K1dJMnRkdk1vV1dSaTlNYXhhUlUveWxPMjRqdmJBNy9QRVo0MEx3PT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

![Table  3:  Overall  F1  score  in  recognizing  different  target  NE  categories  of  the  test  set  of  the  subsequent  step](http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjNadWN1K1dJMnRkbWhJV3hObGg0elBoTzBqb0ZwYUxPQXZOSmlaL1dXM3ZBPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

## Conclusion
本文研究的是迁移学习在序列标注任务上的应用，通过一个adapter来解决领域内迁移的问题。
