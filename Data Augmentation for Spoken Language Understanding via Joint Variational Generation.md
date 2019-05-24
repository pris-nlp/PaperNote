---
title: Data Augmentation for Spoken Language Understanding via Joint Variational Generation
date: 2019-05-23 14:40:00
mathjax: true
tags:
- NLU
- Variational Generation
categories:
- NLP
thumbnail: http://imglf3.nosdn0.126.net/img/bG1jbzEvdHVjVjBmMy9OcER1MjRCTkUrQ0ZnMER2SnROdW5BVy94WXV1cWxVRGUxRDBLeXF3PT0.png?imageView&thumbnail=1680x0&quality=96&stripmeta=0
---
本文定义了数据增强DA的通用结构，并且针对于SLU任务提出了Joint  Language  Understanding  Variational  Au-
toencoder  (JLUVA)模型，在此基础上分析了各种VAE采样的方法。AAAI2019

[paper link](https://drive.google.com/open?id=1XJzN44swGl0NSwXDQ9Jkykjg-ltv_WxP)
conference: AAAI2019
<!-- more -->

## Introduction
标准的SLU任务需要大量的标注数据，本文研究的是基于VAE的SLU数据增强（DA）方法。大部分传统的DA方法只是简单地保留类别信息，对样本进行一定的转译，这类方法需要完整的监督信息，缺乏生成的多样性和鲁棒性。而本文定义了一种基于隐变量的 **generative data augmentation (GDA)**，在多个SLU数据集上进行实验，证明了GDA的有效性。

本文的核心贡献如下：

* 本文定义了一种针对于SLU任务的通用GDA框架  TODO
* 本文提出了一种联合生成utterance和label的生成模型，实验证明可以生成自然的语句，并且可以正确的标注；同时提高了SLU模型的准确率
* 作者通过大量的实验证明本文提出的GDA方法适用于各种SLU数据集和模型。

## Model

### GDA Framework
作者首先描述了SLU任务中GDA的通用框架。

**Notations**  
$w=(w_{1},...,w_{T})$ 是一个utterance，T是这个序列的长度。在一个已标注的SLU数据集中，$s=(s_{1},...,s_{T})$ 是序列w对应的slot标注，序列的意图标注则用y表示。D是一个全部标注过的SLU数据集 $\left\{\left(\mathbf{w}_{1}, \mathbf{s}_{1}, y_{1}\right), \dots,\left(\mathbf{w}_{n}, \mathbf{s}_{n}, y_{n}\right)\right\}$，n是数据集的大小，从D中采样的一个样本为$x=(w,s,y)$，$D_{w}, D_{s}, D_{y}$ 分别代表D中所有utterances、slot labels、intent labels。

**Spoken  Language  Understanding**
本文采用的是slot-intent联合模型，训练损失函数如下：
$$
\mathcal{L}_{L U}(\psi ; \mathbf{w}, \mathbf{s}, y)=-\log p_{\psi}(\mathbf{s}, y | \mathbf{w})
$$

**Generative  Data  Augmentation**
作者从理论上分析了生成式数据增强的通用框架，如Fig 1所示。假设D中的所有样本满足独立同分布，都是从一个真实但未知的语言分布P采样得到 $p(\mathbf{x}) \in \mathcal{P}$，但是由于实际数据收集过程中的偏差，数据集D中的$D_{w}$与真实分布存在差异，定义这种偏差为 $\omega_{b} \in \Omega : \mathcal{P} \rightarrow \mathcal{P^{*}}$，可以使用KL散度来衡量真实分布p与采样分布$p^{*}$。

一个理想的GDA模型应该抵消偏差$\omega_{b}$，通过无监督地采样学习来发掘出真实分布。如果合成的数据完全满足$p^{*}$的分布，那么这种DA的方法不会产生更好的SLU的结果。与之相反，一个好的DA方法应该能得到一个新的分布 $\hat{p}^{\star}=\omega_{d}(\hat{p})$，使得$d\left(\hat{p}^{\star}, p\right)<d(\hat{p}, p)$，即DA采样$w_{d}$能够抵消$w_{b}$的影响。

![](http://imglf5.nosdn0.126.net/img/bG1jbzEvdHVjVjJKZE1EOFF4Q0JqK1hGMjUvNFd3MmI2NVY2aDRISVJTZGlXZ1F0eUhuRnFnPT0.png?imageView&thumbnail=1680x0&quality=96&stripmeta=0)

### Joint  Generative  Model

作者首先将VAE应用到utterance的生成，然后再拓展VAE模型，以一种联合的方式去产生对应的标签信息。

#### Standard  VAE
![](http://imglf5.nosdn.127.net/img/bG1jbzEvdHVjVjBTZUVXb0NjMTdiZk93ZGxYT21WelphVms5REJleXczS2xqRVBZMGlHWm93PT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

**The  Sampling  Problem**
训练好Encoder network和Decoder network后，需要从Decoder network采样得到utterance，即：
$$
\hat{\mathbf{w}} \sim p_{\theta_{\mathcal{D}}, \phi_{\mathcal{D}}}(\mathbf{w})=\int p_{\theta_{\mathcal{D}}}(\mathbf{w} | \mathbf{z}) p_{\theta_{\mathcal{D}}, \phi_{\mathcal{D}}}(\mathbf{z}) d \mathbf{z}
$$
$$
p_{\theta_{\mathcal{D}}, \phi_{\mathcal{D}}}(\mathbf{z})=\mathbb{E}_{\mathbf{w} \sim p(\mathbf{w})}\left[q_{\phi_{\mathcal{D}}}(\mathbf{z} | \mathbf{w})\right]
$$
而w的真实分布往往是未知的，因此需要一些近似的方法来从隐变量中采样，而采样的质量会影响生成样本的好坏。

* VAE中最基础的方法是直接用z的先验分布（标准正态分布）来近似，直接从正态分布中采样z。而这种方法会生成大量同质的和无意义的样本，因为这种假设过于简单。
	> In  real  world  scenarios,  the  KLD  loss  term  in  ELBO loss  is  still  large  after convergence.
* 另一种是基于Monte Carlo的方法
	![](http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjE5V0RNMDA0N1RpRThvWVFHeCswd0JaKy8wNjNzcks3Q2xmM08vdGo2cSt3PT0.png?imageView&thumbnail=1680x0&quality=96&stripmeta=0)
	> According  to  the  law  of  large  numbers,  the  marginal  likelihood $p_{\theta_{\mathcal{D}}, \phi_{\mathcal{D}}}(\mathbf{w})$ converges  to  the  empirical  mean,  thereby providing  an  unbiased  distribution  for  sampling w.

作者还提出了一种**Exploratory  Sampling**的采样方法，目的是增加生成utterance的多样性。作者认为一种理想的采样方法应该是无偏估计，但是方差要增加。假设Algorithm 1中，$\mu, \sigma$ 分别是VAE encoder得到的均值和方差，然后可以从 $\mathcal{N}\left(\boldsymbol{\mu}(\mathbf{w}), \lambda_{s} \cdot \boldsymbol{\sigma}(\mathbf{w})\right)$ 采样z，而参数 $\lambda_{s}$ 用来控制VAE decoder（也叫做generator）探索exploration的程度，影响生成utterance的多样性。

#### Joint  Language  Understanding  VAE
与标准的输入输出均为utterance的VAE模型相比，Joint  Language  Understanding  VAE(JLUVA) 还需要同时预测slot和intent标签，如图2所示：

![](http://imglf3.nosdn0.126.net/img/bG1jbzEvdHVjVjBmMy9OcER1MjRCTkUrQ0ZnMER2SnROdW5BVy94WXV1cWxVRGUxRDBLeXF3PT0.png?imageView&thumbnail=1680x0&quality=96&stripmeta=0)

因此，SLU的loss如下：
$$
\mathcal{L}_{L U}(\phi, \psi ; \mathbf{w}, \mathbf{s}, y)=-\mathbb{E}_{\mathbf{z} \sim q_{\phi}}\left[\log p_{\psi}(\mathbf{s}, y | \hat{\mathbf{w}}, \mathbf{z})\right]
$$
JLUVA的联合loss为：
$$
\begin{aligned} \mathcal{L}(\theta, \phi, \psi ; \mathbf{w}, \mathbf{s}, y)=& \mathrm{D}_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z} | \mathbf{w}) \| p_{\theta}(\mathbf{z} | \mathbf{w})\right) \\ &-\mathbb{E}_{\mathbf{z} \sim q_{\phi}}\left[\log p_{\theta}(\mathbf{w} | \mathbf{z})\right] \\ &-\mathbb{E}_{\mathbf{z} \sim q_{\phi}}\left[\log p_{\psi}(\mathbf{s}, y | \hat{\mathbf{w}}, \mathbf{z})\right] \end{aligned}
$$

> We  obtain  the  optimal  parameters  $\theta^{*}, \phi^{*}, \psi^{*}$ by  minimizing  Equation  6  (i.e.  $\arg \min _{\theta, \phi, \psi} \mathcal{L}$)  with  respect  to  a  real dataset  D.

在数据生成阶段，模型使用某种近似策略（上文Sampling中的方法）来采样，然后通过decoder network $p_{\theta}\left(\mathbf{w} | \mathbf{z}^{\star}\right)$ 来合成utterance $\hat{\mathrm{w}}$，最后再通过SLU网络来预测合成utterance的标签 $\hat{\mathbf{s}}$ 和 $\hat{y}$，三者合为一体得到一个样本$(\hat{\mathbf{w}}, \hat{\mathbf{s}}, \hat{y})$。

## Experiments
### Datasets
作者在以下SLU数据集上做实验：

* ATIS:  Airline  Travel  Information  System  (ATIS) (Hemphill,  Godfrey,  and  Doddington  1990)  is  a  representative  dataset  in  the  SLU  task,  providing  well-founded comparative  environment  for  our  experiments.
* Snips:  The  snips  dataset  is  an  open  source  virtual-assistant  corpus.  The  dataset  contains  user  queries  from various  domains  such  as  manipulating  playlists  or  booking  restaurants.
* MIT  Restaurant  (MR):  This  single-domain  dataset  specializes  in  spoken  queries  related  to  booking  restaurants.
* MIT  Movie:  The  MIT  movie  corpus  consists  of  two single-domain  datasets:  the  movie  eng  (ME)  and  movie trivia  (MT)  datasets.  While  both  datasets  contain  queries about  film  information,  the  trivia  queries  are  more  complex  and  specific.

![](http://imglf3.nosdn0.126.net/img/bG1jbzEvdHVjVjJKditIWDVxdnhPMFNJM3Q2bit0TnYrL0F4NFA5T0JidCt5QTdBTHBTTFhBPT0.png?imageView&thumbnail=1680x0&quality=96&stripmeta=0)

### Experimental  Settings
> Since  we  observe  a  high variance  in  performance  gains  among  different  runs  of  the same  generative  model,  we  need  to  approach the  experimental  designs  with  a  more  conservative  stance.

论文实验设置如下：

* 在相同的训练集下，使用不同的随机数种子来训练$N_{G}$个相同的生成模型
* 从$N_{G}$中每一个模型采样得到m个utterances，得到$N_{G}$个增强后的数据集$\mathcal{D}_{1}^{\prime}, \ldots, \mathcal{D}_{N_{G}}^{\prime}$
* 在每一个数据集上训练$N_{L}$个相同的SLU模型，所有的模型都是在相同的验证集和测试集上评估
* 最终得到$N_{G} \times N_{L}$个结果

### Generative  Data  Augmentation  Results

![](http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjJKditIWDVxdnhPem9OOE5pZ0pIbUhFSGZJZDE4YTdCS1c2S1Y1QWlvR3p3PT0.png?imageView&thumbnail=1680x0&quality=96&stripmeta=0)

实验表明本文提出的方法在小规模数据集上效果提升明显，可能是因为对于大数据集数据增强的意义不大。

**GDA  on  Other  SLU  Models  and  Datasets**
![](http://imglf5.nosdn0.126.net/img/bG1jbzEvdHVjVjJKditIWDVxdnhPeGVoNWh0VGxiL2tiOEhZZkwyMURBa0N3U080MlN6Um9nPT0.png?imageView&thumbnail=1680x0&quality=96&stripmeta=0)

从表3可以看出，本文提出的GDA模型的效果受两个方面因素的影响：

* 数据集本身的难度
* 模型的表达能力

**Comparison  to  Other  State-of-the-art  Results**
![](http://imglf4.nosdn0.126.net/img/bG1jbzEvdHVjVjJKditIWDVxdnhPMG1uTkhGalV0Vis2amMySUswNnJVQUVjSVRxSVhWb3pRPT0.png?imageView&thumbnail=1680x0&quality=96&stripmeta=0)

### Ablation  Studies
作者做了两组消融实验分别来验证采样方法和合成数据比例的影响。

#### Sampling  Methods
1. **Exploratory Monte-Carlo  Posterior  Sampling  (Ours)**:  z  is  sampled from  the  empirical  expectation  of  the  model,  which  is  estimated  by  inferring  posteriors  from  random  utterance  samples.  (Algorithm  1)

2. **Standard  Gaussian**:  z  is  sampled  from  the  assumed prior,  the  standard  multivariate  Gaussian.

3. **Additive  Sampling**:  First,  the  latent  representation  $z_{w}$  of a  random  utterance w  is  sampled.  Then $z_{w}$   is  disturbed  by a  perturbation  vector  α  ∼  U  (−0.2,0.2).  It  was  proposed for  the  deterministic  model  in  (Kurata,  Xiang,  and  Zhou 2016).

实验结果见表2。实验结果表明本文提出的Exploratory Monte-Carlo  Posterior  Sampling是最优的。而简单的Additive  Sampling也取得了不错的效果，表明采样方法并不仅限于高斯分布。最简单的标准正态分布导致模型表现下降，说明这种采样方法有很大的局限性。

#### Synthetic  Data  Ratio
![](http://imglf6.nosdn0.126.net/img/bG1jbzEvdHVjVjJKditIWDVxdnhPNktyQjM2M0M3MS9vSzJxbEpCZ1VoWk9RWXFHZlNuMFhRPT0.png?imageView&thumbnail=1680x0&quality=96&stripmeta=0)

从图3可以看出，在合成数据：原始数据=50左右时，SLU模型的提升效果达到平衡。

## Conclusion
本文定义了数据增强DA的通用结构，并且针对于SLU任务提出了Joint  Language  Understanding  Variational  Au-
toencoder  (JLUVA)模型，在此基础上分析了各种VAE采样的方法。作者最后提到这类方法也可以应用到其它NLP任务中，但是这些工作还需要更多的理论上的解释。

