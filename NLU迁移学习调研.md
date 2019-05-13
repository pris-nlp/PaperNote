## Papers
### 1. [Transfer Learning for Sequence Labeling Using Source Model and Target Data](https://helicqin.github.io/2019/04/12/Transfer%20Learning%20for%20Sequence%20Labeling%20Using%20Source%20Model%20and%20Target%20Data/)

![Figure 2: Our Proposed Neural Adapter](http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjJ5NnVkangrMnRtVE0vSTdHTUw0eGNGQWxjM045K3p1UTh0aVpCaWVWcGtRPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

本文提出了一种渐进式的序列标注模型，模型主要分为两部分：

-   给定在source data  $D_{s}$  训练出的source model  $M_{s}$（实际使用的是Bi-LSTM+CRF），使用其参数来初始化$M_{t}$，同时增加$M_{t}$输出层的维度，然后在target data  $D_{t}$  上fine-tuning。
-   增加了一个neural adapter来连接$M_{s}$和$M_{t}$，通过一个Bi-LSTM来实现，以$M_{s}$的最后线性层输出（未经过softmax）为Bi-LSTM的输入，它的输出作为$M_{t}$的额外输入。适配器adapter的主要作用是解决$D_{s}$和$D_{t}$中标签序列不一致的问题。

> the surface form of a new category type has already appeared in the  DSDS, but they are not annotated as a label. Because it is not yet considered as a concept to be recognized.

#### Related Work

* [Progressive Neural Networks](https://drive.google.com/open?id=1Mti0L5jCKrQXqeIGb8H8cI7I2B6bPkaI): 本文主要解决的是迁移学习中知识遗忘的问题，传统的迁移学习仅仅利用预训练的权重参数然后finetuning，有可能导致之前任务中学到的知识丢失；同时多任务之间可能会存在正交和对抗的关系，简单的预训练权重无法解决。本文提出了一种渐进式学习的框架，核心是通过增加多个任务模型之间横向的连接，来获取之前任务的特征。结构图：

	![Figure  1:  Depiction  of  a  three  column  progressive  network.  The  first  two  columns  on  the  left  (dashed  arrows) were  trained  on  task  1  and  2  respectively.  The  grey  box  labelled  a  represent  the  adapter  layers  (see  text).  A  third column  is  added  for  the  final  task  having  access  to  all  previously  learned  features.](http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjFSYnM4aVcwQ0RtdVIvWWlEelJpQUsrdWVCWTNpVnVqOWM4MGkzcCtFRzFnPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

	上图中的每一列对应一个任务，a代表adapter（这里实际上是MLP），在训练第k个任务时，前k-1个任务的参数是固定的。横向连接如下：

	![](http://imglf5.nosdn0.126.net/img/bG1jbzEvdHVjVjFSYnM4aVcwQ0RtbnJHWGd5Nk1LMXM1SHRjaFh6eXorTG1DdWF4WWprT0xBPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

### 2. [Adversarial Active Learning for Sequences Labeling and Generation](https://helicqin.github.io/2019/04/05/Adversarial%20Active%20Learning%20for%20Sequence%20Labeling%20and%20Generation/)

本文发表在IJCAI2018上，主要是关于active learning在序列问题上的应用，现有的active learning方法大多依赖于基于概率的分类器，而这些方法不适合于序列问题（标签序列的空间太大），作者提出了一种基于adversarial learning的框架解决了该问题。
![Figure 1: An overview of Adversarial Active Learning for sequences (ALISE). The black and blue arrows respectively indicate flows for labeled and unlabeled samples.](http://imglf5.nosdn0.126.net/img/bG1jbzEvdHVjVjJSdFo1UnN1RWJIenIwM0VuKzNsbWw1bjJWcU5xQ2daV2JYVGhUSGZSUGx3PT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

![](http://imglf6.nosdn0.126.net/img/bG1jbzEvdHVjVjFhdEx4eU5Jb3gzS3FoTU9xbDdiVThrUjlFVnl6Y25RTFZkZ1ZRM0dWakl3PT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

与GAN类似，训练过程主要分两步：

1.  Encoder&&Decoder：Mathematically, it encourages the discriminator D to output a score 1 for both  $z_{L}$  and  $z_{U}$.  
    ![](http://imglf3.nosdn0.126.net/img/bG1jbzEvdHVjVjFhdEx4eU5Jb3gzQXE3SUtGQmRmWm5oQWo1d3M3N0xDcVUvS3hnam15VXRnPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)
2.  Discriminator:  
   ![](http://imglf6.nosdn0.126.net/img/bG1jbzEvdHVjVjFhdEx4eU5Jb3gzQlpUQW5vbURFMnBLeDRYR2h3UWJvRHpnbFgwV0dqd3RBPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

> Therefore, the score from this discriminator already serves as an informativeness similarity score that could be directly used for Eq.7.

训练完成之后，将所有的未标注数据通过M和D，来获得匹配度：  

[![](http://imglf3.nosdn0.126.net/img/bG1jbzEvdHVjVjFhdEx4eU5Jb3gzS3FPaXFkcjZJa2FFa3VtTWU0TGlNM3RLUGFVbnEwZlNRPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)](http://imglf3.nosdn0.126.net/img/bG1jbzEvdHVjVjFhdEx4eU5Jb3gzS3FPaXFkcjZJa2FFa3VtTWU0TGlNM3RLUGFVbnEwZlNRPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

> Apparently, those samples with lowest scores should be sent out for labeling because they carry most valuable information in complementary to the current labeled data.

#### Related Work

* [NLP领域的对抗式方法综述](https://github.com/Helicqin/hangtiansuo/wiki/NLP%E9%A2%86%E5%9F%9F%E7%9A%84%E5%AF%B9%E6%8A%97%E5%BC%8F%E6%96%B9%E6%B3%95%E7%BB%BC%E8%BF%B0)

### 3. [Zero-Shot Adaptive Transfer for Conversational Language Understanding](https://pris-nlp.github.io/PaperNote/Zero-Shot%20Adaptive%20Transfer%20for%20Conversational%20Language%20Understanding)

本文提出的模型**Zero-Shot Adaptive Transfer model (ZAT)**借鉴于zero-shot learning，传统的序列标注任务把slot类型作为预测输出，而本文中的模型则是将slot描述信息作为模型输入，如下图：

![Figure  1:  (a)  Traditional  slot  tagging  approaches  with  the BIO  representation.  (b)  For  each  slot,  zero-shot  models  independently  detect  spans  that  contain  values  for  the  slot.  Detected  spans  are  then  merged  to  produce  a  final  prediction.](http://imglf6.nosdn0.126.net/img/bG1jbzEvdHVjVjJsZ0c3RWEvNDlzaTBmUlI4bGdRTnpLKy9oek5CYTFaenBRVnBsVStKZGZ3PT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

针对于同一个utterance，需要独立的经过每一类slot type模型预测结果，之后再把结果合并得到最终的输出。作者假设，不同的领域可以共享slot描述的语义信息，基于此，我们可以在大量的源数据中训练源模型，之后在少量的目标数据上finetune，并且不需要显式地slot对齐。

![Figure 2: Network architecture for the Zero-Shot Adaptive Transfer model.](http://imglf5.nosdn0.126.net/img/bG1jbzEvdHVjVjIxUDhYb2xFNHY2YnhEV1NVRjRoaTdoeFhwZ3VCSk1yS2NwVUUrSEF5Q1BnPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

#### Related Work
* [Frustratingly Easy Neural Domain Adaptation](https://www.aclweb.org/anthology/C16-1038): 本文研究的是是NLU领域迁移问题，属于data-driven的方法。核心是将多个领域的序列标注任务当作多任务学习，然后进行联合训练。缺点是训练集的增大导致训练时间变长。conference: COLING2016

	model architecture:
	![](http://imglf3.nosdn0.126.net/img/bG1jbzEvdHVjVjFhYjl4UytNeldWNnVzVmJWOVZjS002REk4ZjVuMlc5clRuQkFvMUV4aDN3PT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

	上图是两种变体，$x_{t}^{(k)}$代表手工构造的领域相关的特征，$x_{t}$代表领域无关的词向量。k代表第k个领域，$\theta$代表多个领域共享参数，$\theta^{(k)}$代表第k个领域所独有的参数：
	$$
	z_{t}^{k} = \left[W W^{k}\right] \left[ \begin{array}{l}{h_{t}} \\ {h_{t}^{k}}\end{array}\right]=W h_{t}+W^{k} h_{t}^{k} , \text{for figure (a)}\\
	z_{t}^{k} = W x_{t}+W^{k} \theta^{k}\left(x_{t}\right) , \text{for figure (b)} 
	$$

* [Domain Attention with an Ensemble of Experts](https://drive.google.com/open?id=1SZoSEX79Z1Zi9Gml2Hqg3ao23mbPeuDb): 本文研究的是是NLU领域迁移问题，属于model-driven的方法。与data-driven将多个领域的序列标注任务当作多任务学习，然后进行联合训练的方法不同的是，model-driven的方法是将在已有的源数据集训练好的源模型作为额外的特征提取器，然后在训练target model时添加进去，本质上target data并没有增加，因此训练时间远远优于data-driven。conference: ACL2017

	individual model architecture:
	![Figure 1: The overall network architecture of the individual model.](http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjJDWXVGWldNWTZuYmxMNmZLT3Ywa1hqa1NzVVArdURFUU8zVU9KSFZnMU13PT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

	domain transfer architecture:
	![Figure  2:  The  overall  network  architecture  of  the domain  attention,  which  consists  of  three  components:  (1)  K  domain  experts  +  1  target  BiLSTM layer  to  induce  a  feature  representation,  (2)  K  domain  experts  +  1  target  feedfoward  layer  to  output  pre-trained  label  embedding  (3)  a  final  feedforward  layer  to  output an  intent  or  slot.  We  have  two separate  attention  mechanisms  to  combine  feedback  from  domain experts.](http://imglf3.nosdn0.126.net/img/bG1jbzEvdHVjVjJDWXVGWldNWTZuVEg3bFljdC9XRHFadC8wLzlNRzBQeTBGTklCQmR1VGhRPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

	上图实际上对应两种变体：一是利用底层BiLSTM计算注意力作为额外特征，另一种是利用高层label embedding计算注意力作为额外特征。

### 4. [Improving Domain Adaptation Translation with Domain Invariant and Specific Information](https://drive.google.com/openid=1i9I3eKzBbPLanMtf2fVNXxy25BFqTJBh)

![](http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjIySTl1ek9GMmVaWk04K01obFBvTUlHZThCZWNGamI2YlVyV2tDWG54QjRnPT0.jpg?imageView&thumbnail=1680x0&quality=96&stripmeta=0&type=jpg)

训练部分采用了不同的方式，需要关注。

## Insight

