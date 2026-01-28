本仓库包含深度学习资料和一些领域的经典论文，适合初学者入门，请大家先掌握基础深度学习知识，并阅读相关论文，然后再读细分领域的论文

# 基础学习视频

本节主要包括深度学习学习视频和路线，主要包含各个深度学习UP主的讲解视频，感谢他们的付出。学习路线上建议先学习基础，通用的深度学习知识，再聚焦细分领域。

## 基础深度学习

这部分主要参考UP主李沐的视频 [link](https://space.bilibili.com/1567748478/lists/358497?type=series)，其中重要部分如下：

1.安装(请使用pytorch)

2.数据操作+预处理

3.线性代数

4.矩阵计算

5.自动求导

6.线性回归+基础优化算法

7.Softmax回归+损失函数+图片分类数据集

8.多层感知机

9.模型选择+拟合问题

10.权重衰退

11.丢弃法

12.数值稳定性+模型初始化

13.pytorch神经网络基础

13.卷积层

14.卷积层的填充和步骤

15.卷积层的多输入和多输出通道

16.池化层

17.AlexNet

18.批量归一化

19.ResNet 请结合UP霹雳吧啦Wz的视频 [link](https://space.bilibili.com/18161609/lists/244158?type=series)中关于ResNet的代码讲解一起学习

20.数据增广

21.微调

22.全连接神经网络FCN

23.Transformer (此处是NLP中的模型) 请结合UP霹雳吧啦Wz的视频 [link](https://space.bilibili.com/18161609/lists/244158?type=series)中关于自注意力和VIT的代码讲解一起学习

## 代码实战学习

这部分主要参考UP霹雳吧啦Wz的视频 [link](https://space.bilibili.com/18161609/upload/video)，环境请使用pytorch，其中重要部分如下：

1.卷积网络基础

2.AlexNet网络结构详解与花分类数据集下载

3.卷积神经网络基础补充

4.ResNet网络结构，BN以及迁移学习详解

5.使用pytorch搭建ResNet并基于迁移学习训练

6.使用pytorch查看中间层特征矩阵以及卷积核参数

7.Transformer中Self-Attention以及Multi-Head Attention详解

8.Vision Transformer(vit)网络详解

9.使用pytorch搭建Vision Transformer(vit)模型

10.膨胀卷积(Dilated convolution)详解

11.Grad-CAM简介

## 3D视觉和视觉定位基础学习

请学习鲁鹏老师的深入浅出SFM和SLAM核心算法系列 [link](https://www.bilibili.com/video/BV1DP41157dB/?spm_id_from=333.1387.favlist.content.click&vd_source=3792ab8209f08ad74ee4ed2e6d3c812d)

# 论文列表

## Graph Neural Network

Semi-supervised classification with graph convolutional networks. ICLR17

Graph attention networks. Arxiv17

## Pose Estimation

FRAME: Floor-aligned Representation for Avatar Motion from Egocentric Video. CVPR25

3D human pose estimation with spatial and temporal transformers. ICCV21

PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model. AAAI25

## Multimodal

Learning Transferable Visual Models From Natural Language Supervision. ICML21

## Visual Localization

From Coarse to Fine: Robust Hierarchical Localization at Large Scale. CVPR19

Map-Relative Pose Regression for Visual Re-Localization. CVPR24

Map-free Visual Relocalization: Metric Pose Relative to a Single Image. ECCV22

A Survey on Monocular Re-Localization: From the Perspective of Scene Map Representation. TIV24

Long-term Visual Localization with Mobile Sensors. CVPR23

## Panoramic Visual Localization

LDL: Line Distance Functions for Panoramic Localization. ICCV23

PICCOLO: Point Cloud-Centric Omnidirectional Localization. ICCV21

Fully Geometric Panoramic Localization. CVPR24

## Visual Place Recognition

BoQ: A Place is Worth a Bag of Learnable Queries. CVPR24 

Deep Homography Estimation for Visual Place Recognition. AAAI24 

CricaVPR: Cross-image Correlation-aware Representation Learning for Visual Place Recognition. CVPR24 

EDTformer: An Efficient Decoder Transformer for Visual Place Recognition. TCSVT25

TeTRA-VPR: A Ternary Transformer Approach for Compact Visual Place Recognition. RAL25

SuperVLAD: Compact and Robust Image Descriptors for Visual Place Recognition. Neurips24

Optimal Transport Aggregation for Visual Place Recognition. CVPR24

Towards Seamless Adaptation of Pre-trained Models for Visual Place Recognition. ICLR24

MixVPR: Feature Mixing for Visual Place Recognition. WACV23

Focus on Local: Finding Reliable Discriminative Regions for Visual Place Recognition. AAAI25

EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition. ICCV23

Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era. Neurips25

MeshVPR: Citywide Visual Place Recognition Using 3D Meshes. ECCV24

Rethinking Visual Geo-localization for Large-Scale Applications. CVPR22

AnyLoc: Towards Universal Visual Place Recognition. ICRA24
 
## General Visual Tasks

OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels. CVPR25

An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR21

LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference. ICCV21

Is Space-Time Attention All You Need for Video Understanding? ICML21

Masked Autoencoders Are Scalable Vision Learners. CVPR22

Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV21.

Not all patches are what you need: Expediting vision transformers via token reorganization. ICLR22

Vision Transformer with Deformable Attention. CVPR22

Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model.ICML24

MambaOut: Do We Really Need Mamba for Vision? CVPR25

Vision Transformers Need Registers. ICLR24

Segment Anything. ICCV23

Deep residual learning for image recognition. CVPR16

## Local Feature Macth

SFD2: Semantic-guided Feature Detection and Description. CVPR23

DKM: Dense Kernelized Feature Matching for Geometry Estimation. CVPR23

RoMa: Robust Dense Feature Matching. CVPR24.

Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed. CVPR24

EDM: Efficient Deep Feature Matching. ICCV25

CoMatch: Dynamic Covisibility-Aware Transformer for Bilateral Subpixel-Level Semi-Dense Image Matching. ICCV25

JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba. CVPR25

LoFTR: Detector-Free Local Feature Matching with Transformers. CVPR21

SuperGlue: Learning Feature Matching with Graph Neural Networks. CVPR20

## Interpretability

Grad-cam: Visual explanations from deep networks via gradient-based localization. ICCV17

## Visual Foundation Models

VGGT: Visual Geometry Grounded Transformer. CVPR25

DINOv2: Learning Robust Visual Features without Supervision. TMLR24

DINOv3. Arxiv25

## Face Recognition

FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR15

ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR19

## Diffusion 
Do Text-free Diffusion Models Learn Discriminative Visual Representations? ECCV24

Diffusion Models and Representation Learning: A Survey. TPAMI23

## Knowledge Distillation

Improving Language Model Distillation through Hidden State Matching. ICLR25

Rethinking Centered Kernel Alignment in Knowledge Distillation. IJCAI24

Single teacher, multiple perspectives: Teacher knowledge augmentation for enhanced knowledge distillation. ICLR25

## LLM

Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free. Neurips25

## Others

Attention Is All You Need. Neurips17

Mamba: Linear-Time Sequence Modeling with Selective State Spaces. COLM24.






