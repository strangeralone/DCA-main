"""DCA 核心训练模块

包含 DCA 和 DCA+CLIP 训练器的实现。

Classes:
    DCATrainer: 纯 DCA 双分类器适应训练器
    DCAClipTrainer: DCA + CLIP 引导的训练器
"""

from core.dca import DCATrainer
from core.dca_clip import DCAClipTrainer

__all__ = ['DCATrainer', 'DCAClipTrainer']
