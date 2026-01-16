"""DCA 工具模块

包含网络定义、损失函数、数据加载和配置工具。

Modules:
    network: 网络架构定义（ResNet backbone, 双分类器等）
    loss: 损失函数（交叉熵、熵损失、KL散度等）
    data_list: 数据集类
    helpers: 评估、伪标签生成等辅助函数
    config: 配置文件加载工具
"""

from utils.network_new import ResBase, bottleneck, classifier_C, classifier_D
from utils.loss import (
    CrossEntropyLabelSmooth, 
    Entropy, 
    entropy, 
    adentropy, 
    class_balance, 
    SKL, 
    mixup_data
)
from utils.data_list import ImageList, ImageList_idx
from utils.helpers import (
    cal_acc, 
    cal_acc_easy, 
    obtain_label, 
    obtain_label_easy,
    cosine_similarity
)
from utils.config import load_config, get_arg_parser, merge_args_to_config

__all__ = [
    # 网络
    'ResBase', 'bottleneck', 'classifier_C', 'classifier_D',
    # 损失
    'CrossEntropyLabelSmooth', 'Entropy', 'entropy', 'adentropy', 'class_balance', 'SKL', 'mixup_data',
    # 数据
    'ImageList', 'ImageList_idx',
    # 辅助
    'cal_acc', 'cal_acc_easy', 'obtain_label', 'obtain_label_easy', 'cosine_similarity',
    # 配置
    'load_config', 'get_arg_parser', 'merge_args_to_config',
]
