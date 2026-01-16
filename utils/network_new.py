"""
DCA网络架构定义文件
论文标题：Dual Classifier Adaptation: Source-Free UDA via Adaptive Pseudo-labels Learning
此文件实现了DCA方法中的核心网络组件，包括：
1. ResNet特征提取器
2. 双分类器结构（源分类器和目标分类器）
3. 瓶颈层和梯度反转层等辅助模块
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable, Function
from utils import *
import math, pdb
import torch.nn.utils.weight_norm as weightNorm


class GradientReverseFunction(Function):
    """
    梯度反转函数类
    用于实现对抗训练中的梯度反转操作
    在前向传播时保持输入不变，在反向传播时将梯度乘以负系数
    这是实现域对抗训练的关键组件
    """

    @staticmethod
    def forward(ctx, x, coeff):
        """
        前向传播：直接返回输入x
        Args:
            ctx: 上下文对象，用于存储反向传播需要的信息
            x: 输入张量
            coeff: 梯度反转系数
        """
        ctx.coeff = coeff  # 保存系数用于反向传播
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：将梯度乘以负系数实现梯度反转
        Args:
            ctx: 上下文对象
            grad_output: 从上层传来的梯度
        Returns:
            反转后的梯度和None（对应coeff参数的梯度）
        """
        output = grad_output * -ctx.coeff  # 梯度反转
        return output, None


def grad_reverse(x, coeff=1.0):
    """
    梯度反转函数的调用接口
    Args:
        x: 输入张量
        coeff: 梯度反转系数，默认为1.0
    Returns:
        经过梯度反转层的输出
    """
    return GradientReverseFunction.apply(x, coeff)


def init_weights(m):
    """
    网络层权重初始化函数
    根据不同层类型采用不同的初始化策略：
    - 卷积层：使用Kaiming均匀分布初始化
    - 批归一化层：权重使用正态分布初始化，偏置置零
    - 线性层：使用Xavier正态分布初始化

    Args:
        m: 神经网络模块
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        # 卷积层和转置卷积层的初始化
        nn.init.kaiming_uniform_(m.weight)  # Kaiming初始化适合ReLU激活函数
        nn.init.zeros_(m.bias)  # 偏置置零
    elif classname.find('BatchNorm') != -1:
        # 批归一化层的初始化
        nn.init.normal_(m.weight, 1.0, 0.02)  # 权重使用均值1.0，标准差0.02的正态分布
        nn.init.zeros_(m.bias)  # 偏置置零
    elif classname.find('Linear') != -1:
        # 线性层（全连接层）的初始化
        nn.init.xavier_normal_(m.weight)  # Xavier初始化保持梯度方差稳定
        nn.init.zeros_(m.bias)  # 偏置置零


# ResNet模型字典映射
# 支持多种ResNet变体，用于构建特征提取器backbone
res_dict = {
    "resnet18": models.resnet18,     # ResNet-18
    "resnet34": models.resnet34,     # ResNet-34
    "resnet50": models.resnet50,     # ResNet-50（论文中主要使用）
    "resnet101": models.resnet101,   # ResNet-101（VisDA-C数据集使用）
    "resnet152": models.resnet152,   # ResNet-152
    "resnext50": models.resnext50_32x4d,    # ResNeXt-50
    "resnext101": models.resnext101_32x8d,  # ResNeXt-101
}


class ResBase(nn.Module):
    """
    ResNet特征提取器基类
    论文中的特征提取器G，用于从输入图像中提取深层特征
    移除了ResNet的最后一层分类器，只保留特征提取部分
    这是DCA方法中共享的特征提取器，同时为源分类器和目标分类器提供特征
    """

    def __init__(self, res_name):
        """
        初始化ResNet特征提取器
        Args:
            res_name: ResNet模型名称（如'resnet50', 'resnet101'等）
        """
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)  # 加载预训练的ResNet模型

        # 提取ResNet的各个组件（除了最后的全连接层）
        self.conv1 = model_resnet.conv1      # 第一个卷积层 (7x7, stride=2)
        self.bn1 = model_resnet.bn1          # 批归一化层
        self.relu = model_resnet.relu        # ReLU激活函数
        self.maxpool = model_resnet.maxpool  # 最大池化层 (3x3, stride=2)
        self.layer1 = model_resnet.layer1    # ResNet第1个残差块组
        self.layer2 = model_resnet.layer2    # ResNet第2个残差块组
        self.layer3 = model_resnet.layer3    # ResNet第3个残差块组
        self.layer4 = model_resnet.layer4    # ResNet第4个残差块组
        self.avgpool = model_resnet.avgpool  # 全局平均池化层
        self.in_features = model_resnet.fc.in_features  # 特征维度（ResNet50为2048）

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像张量 (batch_size, 3, H, W)
        Returns:
            特征向量 (batch_size, feature_dim) 其中feature_dim=2048 for ResNet50
        """
        x = self.conv1(x)       # 第一个卷积层
        x = self.bn1(x)         # 批归一化
        x = self.relu(x)        # ReLU激活
        x = self.maxpool(x)     # 最大池化
        x = self.layer1(x)      # 残差块组1
        x = self.layer2(x)      # 残差块组2
        x = self.layer3(x)      # 残差块组3
        x = self.layer4(x)      # 残差块组4
        x = self.avgpool(x)     # 全局平均池化
        x = x.view(x.size(0), -1)  # 展平为向量 (batch_size, feature_dim)
        return x


class bottleneck(nn.Module):
    """
    瓶颈层（Bottleneck Layer）
    用于降低特征维度，连接在特征提取器和源分类器之间
    论文中提到源分类器通过瓶颈层连接，而目标分类器直接连接到特征提取器
    这种设计使得两个分类器有不同的感受野，从而产生预测差异
    """

    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        """
        初始化瓶颈层
        Args:
            feature_dim: 输入特征维度（如ResNet50的2048）
            bottleneck_dim: 瓶颈层输出维度，默认256
            type: 瓶颈层类型，"ori"为原始版本，"bn"包含批归一化
        """
        super(bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.dropout = nn.Dropout(p=0.5)   # Dropout正则化，防止过拟合
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)  # 线性变换降维
        self.bottleneck.apply(init_weights)  # 应用权重初始化
        self.type = type

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征 (batch_size, feature_dim)
        Returns:
            降维后的特征 (batch_size, bottleneck_dim)
        """
        x = self.bottleneck(x)  # 线性变换降维
        if self.type == "bn":
            x = self.bn(x)      # 如果类型为"bn"，应用批归一化
        return x


class classifier_C(nn.Module):
    """
    源分类器（Source Classifier）- 论文中的C
    这是在源域预训练后被冻结的分类器，保留源域的判别知识
    通过瓶颈层连接，具有较深的感受野
    在目标域适应过程中保持固定，作为源域知识的载体
    """

    def __init__(self, class_num, bottleneck_dim=256, type="dca"):
        """
        初始化源分类器
        Args:
            class_num: 类别数量
            bottleneck_dim: 瓶颈层特征维度，默认256
            type: 分类器类型，默认"dca"
        """
        super(classifier_C, self).__init__()
        self.type = type
        if type == "dca":
            # 两层全连接网络结构，使用权重归一化
            self.fc1 = weightNorm(nn.Linear(bottleneck_dim, bottleneck_dim), name="weight")
            self.fc1.apply(init_weights)
            self.fc2 = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc2.apply(init_weights)

    def forward(self, x, reverse=False, coeff=0.1):
        """
        前向传播
        Args:
            x: 输入特征 (batch_size, bottleneck_dim)
            reverse: 是否应用梯度反转，默认False
            coeff: 梯度反转系数，默认0.1
        Returns:
            分类预测 (batch_size, class_num)
        """
        x = self.fc1(x)         # 第一层全连接
        if reverse:
            x = grad_reverse(x, coeff)  # 如果需要，应用梯度反转（用于对抗训练）
        x = self.fc2(x)         # 第二层全连接，输出类别预测
        return x


class classifier_D(nn.Module):
    """
    目标分类器（Target Classifier）- 论文中的C*
    这是新引入的目标分类器，直接连接到特征提取器
    在目标域适应过程中持续训练，学习目标域的特定知识
    与源分类器配合，通过输出差异来评估伪标签的可靠性
    """

    def __init__(self, feature_dim, class_num, type="dca"):
        """
        初始化目标分类器
        Args:
            feature_dim: 输入特征维度（直接来自特征提取器，如2048）
            class_num: 类别数量
            type: 分类器类型，默认"dca"
        """
        super(classifier_D, self).__init__()
        self.type = type
        if type == "dca":
            # 两层全连接网络结构，使用权重归一化
            # 注意：这里没有使用批归一化、ReLU或Dropout（被注释掉）
            # self.bn = nn.BatchNorm1d(feature_dim, affine=True)
            # self.relu = nn.ReLU(inplace=True)
            # self.dropout = nn.Dropout(p=0.5)
            self.fc1 = weightNorm(nn.Linear(feature_dim, feature_dim), name="weight")
            self.fc1.apply(init_weights)
            self.fc2 = weightNorm(nn.Linear(feature_dim, class_num), name="weight")
            self.fc2.apply(init_weights)

    def forward(self, x, reverse=False, coeff=0.1):
        """
        前向传播
        Args:
            x: 输入特征 (batch_size, feature_dim)
            reverse: 是否应用梯度反转，默认False
            coeff: 梯度反转系数，默认0.1
        Returns:
            分类预测 (batch_size, class_num)
        """
        x = self.fc1(x)         # 第一层全连接
        if reverse:
            x = grad_reverse(x, coeff)  # 如果需要，应用梯度反转
        x = self.fc2(x)         # 第二层全连接，输出类别预测
        return x


class generator(nn.Module):
    """
    生成器网络（可选组件）
    网络架构与infoGAN相同 (https://arxiv.org/abs/1606.03657)
    注意：这个生成器在DCA论文的主要方法中并未使用
    主要用于一些基于生成的域适应方法的对比实验
    """

    def __init__(self, input_dim=100, input_size=224, class_num=10, batch_size=64):
        """
        初始化生成器
        Args:
            input_dim: 输入噪声维度，默认100
            input_size: 生成图像尺寸，默认224
            class_num: 类别数量，默认10
            batch_size: 批处理大小，默认64
        """
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        self.class_num = class_num
        self.batch_size = batch_size

        # 标签嵌入层：将类别标签转换为嵌入向量
        self.label_emb = nn.Embedding(self.class_num, self.input_dim)

        # 全连接层：将噪声向量映射到特征图
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),  # 第一层全连接
            nn.BatchNorm1d(1024),             # 批归一化
            nn.ReLU(),                        # ReLU激活
            # 第二层全连接：映射到卷积特征图尺寸
            nn.Linear(1024, 128 * (self.input_size // 16) * (self.input_size // 16)),
            nn.BatchNorm1d(128 * (self.input_size // 16) * (self.input_size // 16)),
            nn.ReLU(),
        )

        # 转置卷积层：上采样生成图像特征
        self.deconv = nn.Sequential(
            # 第一个转置卷积层
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 第二个转置卷积层
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 注释掉的最后一层可能用于直接生成RGB图像
            # nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            # nn.Tanh(),
        )
        init_weights(self)  # 应用权重初始化

    def forward(self, input, label):
        """
        前向传播
        Args:
            input: 输入噪声向量 (batch_size, input_dim)
            label: 类别标签 (batch_size,)
        Returns:
            生成的特征向量（展平后）
        """
        # 将标签嵌入与输入噪声相乘（条件生成）
        x = torch.mul(self.label_emb(label), input)
        # 也可以使用拼接的方式：x = torch.cat([input, label], 1)

        x = self.fc(x)  # 通过全连接层
        # 重塑为4D张量用于转置卷积
        x = x.view(-1, 512, (self.input_size // 32), (self.input_size // 32))
        x = self.deconv(x)  # 通过转置卷积层
        x = x.view(x.size(0), -1)  # 展平为向量输出

        return x