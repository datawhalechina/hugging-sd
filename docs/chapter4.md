# 第4章 扩散模型

> The universe was born from chaos, and order emerges from disorder.（宇宙从混沌中诞生，秩序从无序中显现）

&emsp;&emsp;扩散模型（Diffusion Models）是生成模型中一种新兴且备受关注的框架，它通过逐步引入和移除噪声的方式学习数据分布，从而生成与输入数据相似的新样本。这种模型在图像生成、音频建模和视频生成等领域表现出色，展现了优异的生成质量和灵活性。扩散模型的核心思想是构建一个前向扩散过程，将数据逐渐转化为纯噪声，同时通过逆向扩散过程学习将噪声还原为数据的能力。基于这种逐步生成的方式，扩散模型能够更精细地控制生成过程，提高生成样本的质量和多样性。近年来，扩散模型的变体如DDPMs和SGMs在生成任务中取得了显著进展。DDPM通过概率建模优化逆向扩散过程，使得模型能够生成高保真数据，而SGMs则通过学习噪声数据的梯度场（即得分函数）实现生成样本的高效推断。尽管扩散模型的训练过程通常计算密集，但其生成过程可以通过优化和改进大幅加速。扩散模型因其优异的表现成为了生成模型研究的热点，不仅推动了生成任务在分辨率和真实性上的突破，还为多模态数据生成、无监督学习等领域带来了新的可能。在后续部分，我们将详细探讨扩散模型的基本原理、前向与逆向扩散过程，以及它们如何通过逐步逼近数据分布，开辟深度生成模型的新方向。

## 4.1 数据基础

在这一小节中，主要介绍理解扩散模型所需要的基础数学知识。数学上，假设A、B表示两个事件发生的概率，则条件概率的乘法公式一般形式是：

$$
P(A)>0, P(AB)=P(A)P(B|A)
$$

推广到多个事件：

$$
\begin{aligned}
P(A,B,C)  &= P(C|B,A)P(B,A) = P(C|B,A)P(B|A)P(A)\\
P(B,C|A)  &= P(B|A)P(C|A,B)
\end{aligned}
$$

其中是关于条件概率、联合概率的计算公式。在进一步引入基于马尔科夫假设的条件概率计算公式之前，我们需要线了解下什么是马尔可夫性？

马尔可夫性是指给定过去的状态$A=\left \{ X_{0},X_{1},...,X_{n-1}  \right \}$和现在的状态$B=X_{n}$，将来的状态的条件分布$C=X_{n+1}$与过去的状态独立，只依赖于现在的状态。所以基于马尔科夫假设的条件概率计算公式为：

$$
\begin{aligned}
P(A,B,C) &= P(C|B)P(B|A)P(A)\\
P(B,C|A) &= P(B|A)P(C|B)
\end{aligned}
$$

可以看出，马尔科夫假设的条件概率计算公式主要是针对条件概率的条件化约束。

接下来简单介绍KL散度公式，在统计学意义上来说，KL散度可以用来衡量两个分布之间的差异程度。若两个分布差异越小，KL散度越小，反之亦反。当两分布一致时，其KL散度为0。正是因为其可以衡量两个分布之间的差异，所以在VAE、EM、GAN中均有使用到KL散度。假设$p$和$q$均是服从正态分布$N(\mu 1,\rho 1)$和$N(\mu 2,\rho 2)$的随机变量的概率密度函数，则从$p$和$q$的KL散度定义为：

$$
KL(p\left |  \right |q)=\int [log(p(x))-log(q(x))]p(x)dx=\int [ p(x)log(p(x))-p(x)log(q(x))]dx
$$

继续推导为：

$$
KL(p,q)=log\frac{\rho 2}{\rho 1}+\frac{\rho 1^{2}+(\mu1-\mu 2)^2 }{2\rho 2^{2}} -\frac{1}{2}
$$


## 4.2 加噪与去噪

去噪扩散概率模型（Denoising Diffusion Probabilistic Models，DDPM）是一种基于扩散过程的生成模型，通过逐步添加和去除噪声来生成数据。DDPM的核心是通过模拟数据的扩散（加噪）和逆扩散（去噪）过程来生成样本。其灵感来源于热力学中的扩散现象，通过两个马尔可夫链过程实现：

* **正向过程（扩散过程）**：逐步向数据添加高斯噪声，直至数据变为纯噪声。
* **反向过程（去噪过程）**：学习从噪声中逐步恢复原始数据。

定义一个真实数据分布中采样的点$\mathbf{x}_0 \sim q(\mathbf{x})$，正向过程则是逐渐在这个采样中加入$T$次高斯噪声，这形成一系列高斯噪声采样$\mathbf{x}_1, \dots, \mathbf{x}_T$，其中步长由变量$\{\beta_t \in (0, 1)\}_{t=1}^T$控制。数学描述为：

$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$

其中$q(\mathbf{x}_t \vert \mathbf{x}_0)= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$，通过重参数化技巧，可直接从$x_0$计算任意时间步$t$的噪声数据$x_t$：

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
   &= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
\end{aligned}
$$

其中，$\alpha_t = \prod_{s=1}^t (1-\beta_s) $，且$ \alpha_t $随$ t $增大而递减，$\epsilon$是噪声。采样数据$\mathbf{x}_0$在$t$逐渐变大的过程中逐渐失去它原有的分布，当$T \to \infty$时，$\mathbf{x}_T$就等价于一个各向异性高斯分布。

上述过程反过来采样$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$，将逐渐地从高斯噪声分布$\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$中逐恢复出真实的数据$x_0$。但是有两点需要注意，首先是如果步长$\beta_t$太小则$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$也还是高斯噪声分布，其次是由于并不知道整体数据的真实分布所以我们不能很简单的估计$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$。因此我们用模型学习一个参数$p_\theta$来估计这个条件概率分布从而来实现这个反向过程。

$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$

值得注意的是可以从$\mathbf{x}_0$很容易地控制这个反向的条件概率过程：

$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; {\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), {\tilde{\beta}_t} \mathbf{I})
$$

原因是：

$$
{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t
$$

其中，$\alpha_t$和$\bar{\alpha}_t$是预定义的噪声调度参数，$\beta_t$是前向过程的噪声方差。由上式可以看到，$\tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0)$ 是 $\mathbf{x}_0$ 和 $\mathbf{x}_t$ 的线性组合，因此如果我们能够准确估计 $\mathbf{x}_0$，就能直接控制反向过程的均值，从而影响整个去噪过程。可以把扩散模型的反向过程想象成一个逐步去噪的过程，而$\mathbf{x}_0$相当于这个过程的“终点”或者“目标”。每一步的去噪都是在向$\mathbf{x}_0$ 逼近，因此如果我们能控制$\mathbf{x}_0$，整个去噪过程就会按照我们希望的方式进行。总结来说，$\mathbf{x}_0$直接决定了反向过程的均值，因此我们可以通过它来控制整个反向扩散过程，使其朝向期望的生成目标发展。

DDPM的目的是需要得到最优的$\theta$，为此优化目标是通过最大化变分下界（ELBO）进行训练，损失函数简化为预测噪声的均方误差：

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

其中，$t$ 从 $ \{1, ..., T\} $均匀采样，$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon $，其中$ \epsilon \sim \mathcal{N}(0, \mathbf{I}) $。$\theta$用于预测噪声的神经网络通常采用U-Net结构，结合残差连接和时间步嵌入。DDPM相比与其他生成模型的优缺点也非常明显：

* 生成质量高，尤其在图像细节和多样性上优于GAN。
* 训练稳定，无需对抗训练，但是生成速度慢，需多次迭代（通常数百至上千步）。
* 理论推导严密，过程可解释强。


## 4.3 架构优化
