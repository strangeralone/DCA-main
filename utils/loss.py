"""
DCA损失函数定义文件
论文标题：Dual Classifier Adaptation: Source-Free UDA via Adaptive Pseudo-labels Learning

此文件实现了DCA方法中使用的各种损失函数，包括：
1. 带标签平滑的交叉熵损失
2. 熵损失（最大化和最小化）
3. 类平衡损失
4. 对称KL散度损失
5. Mixup数据增强
这些损失函数共同实现了论文中的双分类器自适应机制
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import math
import torch.nn.functional as F
import pdb
from torch.distributions.beta import Beta


class GradientReverseFunction(Function):
    """
    梯度反转函数类（与network_new.py中相同）
    用于实现对抗训练中的梯度反转操作
    """

    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * -ctx.coeff
        return output, None


def grad_reverse(x, coeff=1.0):
    """梯度反转函数的调用接口"""
    return GradientReverseFunction.apply(x, coeff)


class CrossEntropyLabelSmooth(nn.Module):
    """
    带标签平滑的交叉熵损失
    参考：Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    标签平滑公式：y = (1 - epsilon) * y + epsilon / K
    其中K是类别数，epsilon是平滑参数

    标签平滑的作用：
    1. 防止模型过度自信，提高泛化能力
    2. 在源域训练阶段使用，帮助模型学习更鲁棒的特征
    """

    def __init__(self, num_classes, epsilon=0.1, device="cuda", size_average=True):
        """
        初始化标签平滑交叉熵损失
        Args:
            num_classes: 类别数量
            epsilon: 平滑参数，默认0.1
            device: 计算设备，如 "cuda"、"mps" 或 "cpu"
            size_average: 是否对batch求平均
        """
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.device = device
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        前向传播
        Args:
            inputs: 模型预测logits (batch_size, num_classes)
            targets: 真实标签 (batch_size,)
        Returns:
            平滑后的交叉熵损失
        """
        log_probs = self.logsoftmax(inputs)  # 计算log概率
        # 将标签转换为one-hot编码
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.device != "cpu":
            targets = targets.to(self.device)

        # 应用标签平滑：(1-ε)*y_true + ε/K
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


def Entropy(input):
    """
    计算熵值
    Args:
        input: softmax概率分布 (batch_size, num_classes)
    Returns:
        每个样本的熵值 (batch_size,)
    """
    epsilon = 1e-5  # 防止log(0)
    entropy = -input * torch.log(input + epsilon)  # -p*log(p)
    entropy = torch.sum(entropy, dim=1)  # 对类别维度求和
    return entropy


def adentropy(f, feat, lamda, coeff=1.0):
    """
    对抗熵损失（Adversarial Entropy Loss）
    用于对抗训练，通过梯度反转层计算熵损失
    在论文的minimax优化中使用

    Args:
        f: 分类器函数
        feat: 输入特征
        lamda: 损失权重系数
        coeff: 梯度反转系数
    Returns:
        对抗熵损失值
    """
    out_t1 = f(feat, reverse=True, coeff=coeff)  # 通过梯度反转层
    out_t1 = nn.Softmax(dim=1)(out_t1)  # softmax归一化
    # 计算熵损失：λ * mean(sum(p * log(p)))
    loss_adent = lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent


def entropy(f, feat, lamda):
    """
    标准熵损失
    用于最大化预测的不确定性，鼓励模型输出均匀分布
    在目标分类器训练中使用

    Args:
        f: 分类器函数
        feat: 输入特征
        lamda: 损失权重系数
    Returns:
        熵损失值（取负号进行最大化）
    """
    out_t1 = f(feat)  # 分类器输出
    softmax_out = nn.Softmax(dim=1)(out_t1)  # softmax归一化
    # 计算负熵损失：-λ * mean(sum(p * log(p)))
    loss_ent = lamda * torch.mean(torch.sum(-softmax_out * (torch.log(softmax_out + 1e-5)), 1))
    return loss_ent


def class_balance(input, lamda):
    """
    类平衡损失
    论文公式(6)对应的L_cb损失
    通过最小化平均预测分布的熵来鼓励类别平衡
    防止模型将所有样本预测为某个优势类别

    Args:
        input: 模型预测的softmax概率 (batch_size, num_classes)
        lamda: 损失权重系数
    Returns:
        类平衡损失值
    """
    msoftmax = input.mean(dim=0)  # 计算批次内的平均预测分布
    # 最小化平均分布的熵：λ * sum(p_avg * log(p_avg))
    loss_div = lamda * torch.sum(msoftmax * (torch.log(msoftmax + 1e-5)))
    return loss_div


def SKL(out1, out2):
    """
    对称KL散度（Symmetric KL Divergence）
    论文公式(4)对应的L_skl损失
    用于衡量两个分类器输出分布的差异

    SKL(P||Q) = [KL(P||Q) + KL(Q||P)] / 2

    Args:
        out1: 第一个分类器的logits输出
        out2: 第二个分类器的logits输出
    Returns:
        对称KL散度 (batch_size, num_classes)
    """
    out2_t = out2.clone().detach()  # 停止梯度传播
    out1_t = out1.clone().detach()  # 停止梯度传播

    # 计算双向KL散度并平均
    return (F.kl_div(F.log_softmax(out1, dim=1), out2_t, reduction='none') +
            F.kl_div(F.log_softmax(out2, dim=1), out1_t, reduction='none')) / 2


def mixup_data(images, labels, alpha):
    """
    Mixup数据增强
    论文公式(7-8)对应的mixup操作
    通过线性组合两个样本及其标签来生成新的训练样本
    提高模型的泛化能力和对噪声标签的鲁棒性

    Args:
        images: 输入图像 (batch_size, C, H, W)
        labels: 对应标签 (batch_size, num_classes) - 已经是one-hot或soft标签
        alpha: Beta分布参数，控制混合强度
    Returns:
        mixed_images: 混合后的图像
        mixed_labels: 混合后的标签
    """
    batch_size = images.size(0)
    indices = torch.randperm(batch_size)  # 随机排列索引
    shuffled_images = images[indices]     # 打乱后的图像
    shuffled_labels = labels[indices]     # 打乱后的标签

    # 从Beta分布采样混合比例λ
    lam = Beta(alpha, alpha).sample().to(images.device)
    lam = torch.max(lam, 1 - lam)  # 确保λ >= 0.5，保持主要成分

    # 线性组合图像和标签
    mixed_images = lam.view(-1, 1, 1, 1) * images + (1 - lam).view(-1, 1, 1, 1) * shuffled_images
    mixed_labels = lam * labels + (1 - lam) * shuffled_labels

    return mixed_images, mixed_labels


def iid_loss(x_out, x_tf_out, lamb=1.0, eps=1e-8):
    """
    IID Loss (Invariant Information Clustering Loss)
    论文: DIFO - Source-Free Domain Adaptation with Frozen Multimodal Foundation Model (CVPR 2024)
    
    通过最大化两个分布之间的互信息来训练 Prompt Learner。
    公式: L_IID = -I(X; Y) = -sum(P(x,y) * [log P(x,y) - log P(x) - log P(y)])
    
    Args:
        x_out: CLIP 模型的 softmax 输出 (batch_size, num_classes)
        x_tf_out: 伪标签的 softmax 分布 (batch_size, num_classes)
        lamb: 边际概率的权重系数，默认1.0
        eps: 数值稳定性的小常数
    
    Returns:
        IID 损失值（标量）
    """
    # 确保输入是 2D
    if x_out.dim() == 1:
        x_out = x_out.unsqueeze(0)
        x_tf_out = x_tf_out.unsqueeze(0)
    
    _, k = x_out.size()
    
    # 计算联合概率矩阵 P(CLIP_pred, pseudo_label)
    # p_i_j[i,j] = sum_n(x_out[n,i] * x_tf_out[n,j]) / N
    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # (batch, k, k)
    p_i_j = p_i_j.sum(dim=0)  # (k, k)
    
    # 对称化：P = (P + P^T) / 2
    p_i_j = (p_i_j + p_i_j.t()) / 2.
    
    # 归一化为概率分布
    p_i_j = p_i_j / p_i_j.sum()
    
    # 计算边际概率
    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)  # P(CLIP_pred)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # P(pseudo_label)
    
    # 数值稳定性处理
    p_i_j = torch.clamp(p_i_j, min=eps)
    
    # 互信息损失: -I(X;Y) = -sum(p_ij * [log p_ij - log p_i - log p_j])
    # 这里使用负号是因为我们要最大化互信息，但优化器是最小化
    loss = - p_i_j * (torch.log(p_i_j) 
                      - lamb * torch.log(p_j + eps) 
                      - lamb * torch.log(p_i + eps))
    
    return loss.sum()