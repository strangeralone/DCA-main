"""DCA 核心训练模块

包含 DCA 和 DCA+CLIP 训练器的实现。

Classes:
    DCATrainer: 纯 DCA 双分类器适应训练器
    DCAClipTrainer: DCA + CLIP 引导的训练器
    DCACoOpTrainer: DCA + CoOp 提示学习的训练器
    DCAAblationTrainer: DCA + VLM 消融实验训练器
"""

from core.dca import DCATrainer
from core.dca_clip import DCAClipTrainer
from core.dca_coop import DCACoOpTrainer
from core.dca_ablation import DCAAblationTrainer

__all__ = ['DCATrainer', 'DCAClipTrainer', 'DCACoOpTrainer', 'DCAAblationTrainer']

