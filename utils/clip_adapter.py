"""CLIP Adapter 模块

实现轻量级 Adapter，让 CLIP 更好地适应目标域。

架构：在 CLIP 视觉编码器输出后添加 bottleneck Adapter：
- 输入：CLIP 图像特征 [batch, dim]
- Adapter：Linear(dim, dim/r) → ReLU → Linear(dim/r, dim)
- 输出：残差连接 α * adapter(x) + (1-α) * x

参考：
- CLIP-Adapter: Better Vision-Language Models with Feature Adapters (IJCV 2023)
- Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling (ECCV 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPAdapter(nn.Module):
    """CLIP 视觉 Adapter
    
    在 CLIP 图像编码器输出后添加轻量 Adapter，
    通过目标域数据微调，让 CLIP 更好地适应目标域。
    
    Args:
        in_dim: 输入特征维度（CLIP 输出维度）
        bottleneck_ratio: bottleneck 压缩比，默认 4
        residual_ratio: 残差连接权重 α，默认 0.2
        dropout: dropout 概率，默认 0.1
    """
    
    def __init__(self, in_dim=512, bottleneck_ratio=4, residual_ratio=0.2, dropout=0.1):
        super().__init__()
        
        hidden_dim = in_dim // bottleneck_ratio
        
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim)
        )
        
        self.alpha = residual_ratio
        
        # 初始化：使用较小的权重，让初始输出接近原始特征
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: CLIP 图像特征 [batch, dim]
            
        Returns:
            adapted_x: 经过 Adapter 的特征 [batch, dim]
        """
        # 残差连接：α * adapter(x) + (1-α) * x
        return self.alpha * self.adapter(x) + (1 - self.alpha) * x


class MultiLayerAdapter(nn.Module):
    """多层 Adapter（可选，用于更复杂的适应）
    
    在多个位置添加 Adapter，可实现更细粒度的适应。
    
    Args:
        in_dim: 输入特征维度
        num_layers: Adapter 层数
        bottleneck_ratio: bottleneck 压缩比
        residual_ratio: 残差连接权重
    """
    
    def __init__(self, in_dim=512, num_layers=2, bottleneck_ratio=4, residual_ratio=0.2):
        super().__init__()
        
        self.adapters = nn.ModuleList([
            CLIPAdapter(in_dim, bottleneck_ratio, residual_ratio)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        """前向传播"""
        for adapter in self.adapters:
            x = adapter(x)
        return x


class DualModalAdapter(nn.Module):
    """双模态 Adapter（可选，同时适应视觉和文本）
    
    同时在视觉端和文本端添加 Adapter，实现双模态适应。
    
    Args:
        visual_dim: 视觉特征维度
        text_dim: 文本特征维度（通常与视觉相同）
        bottleneck_ratio: bottleneck 压缩比
        residual_ratio: 残差连接权重
    """
    
    def __init__(self, visual_dim=512, text_dim=512, bottleneck_ratio=4, residual_ratio=0.2):
        super().__init__()
        
        self.visual_adapter = CLIPAdapter(visual_dim, bottleneck_ratio, residual_ratio)
        self.text_adapter = CLIPAdapter(text_dim, bottleneck_ratio, residual_ratio)
    
    def adapt_visual(self, x):
        """适应视觉特征"""
        return self.visual_adapter(x)
    
    def adapt_text(self, x):
        """适应文本特征"""
        return self.text_adapter(x)
    
    def forward(self, visual_features, text_features):
        """同时适应两种特征"""
        return self.adapt_visual(visual_features), self.adapt_text(text_features)
