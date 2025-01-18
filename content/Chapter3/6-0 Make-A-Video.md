# [Make-A-Video](https://arxiv.org/abs/2209.14792)

## 0 章节目标

1. 理解Make-A-Video的主要思想
2. 使用Make-A-Video进行视频的创造

## 问题引言

​	     Make-A-Scene 是一种多模式生成人工智能方法，使人们能够更好地控制他们创建的人工智能生成内容。

​		 该系统使用带有描述的图像来了解世界是什么样子以及通常如何描述它。它还使用未标记的视频来了解世界如何运动。有了这些数据，Make-A-Video 就可以让您通过几句话或几行文本生成异想天开、独一无二的视频，将您的想象力变为现实。

## 原理

![1841a19a1b71bfdb7cb5ba719f29894](./../../../AppData/Local/Temp/WeChat%20Files/1841a19a1b71bfdb7cb5ba719f29894.png)

​		  Make-A-Video由三个主要组件组成：**基于文本图像对训练的基础Text to Image模型**、**扩展网络构建块到时间维度的时空卷积和注意力层**，**以及包含时空层和帧插值网络的时空网络**，用于生成高帧率的Text to Video。

​		Make-A-Video的最终Text to Video推理方案:
$$
\hat{y_{t}}=\mathrm{SR}_{h}\circ\mathrm{SR}_{l}^{t}\circ\uparrow_{F}\circ\mathrm{D}^{t}\circ\mathrm{P}\circ(\hat{x},\mathrm{C}_{x}(x))
$$
​	$\hat{y_{t}}$ 表示生成的视频， $\mathrm{SR}_{h}   \mathrm{SR}_{l}^{t}$ 是空间和时空超分辨率网络 ，${D}^{t}$ 是时空解码器, ${P}$ 是先前的，$\hat{x}$ 是BPE编码文本 ，${C}_{x}(x)$  是CLIP的文本编码



## Make-A-Video的使用

