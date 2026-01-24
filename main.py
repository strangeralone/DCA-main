#!/usr/bin/env python
"""DCA 统一训练入口

支持 DCA 和 DCA+CLIP 两种方法，所有参数通过配置文件和命令行控制。

用法示例:
    # 训练源域模型
    python main.py --method dca --dataset officehome --source 0 --mode source
    
    # 目标域适应（使用 CLIP）
    python main.py --method dca_coop --dataset officehome --source 0 --target 1 --mode target
    
    # 完整流程（源域 + 目标域）
    python main.py --method dca_coop --dataset officehome --source 0 --target 1 --mode all
    
    # 覆盖配置参数
    python main.py --method dca --source 0 --mode source --max_epoch 5 --batch_size 32
"""

import os
import random
import numpy as np
import torch

from utils.config import load_config, get_arg_parser, merge_args_to_config


def set_seed(seed: int):
    """设置随机种子以确保可复现性
    
    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    # 解析命令行参数
    parser = get_arg_parser()
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(
        method=args.method,
        dataset=args.dataset,
        config_dir=args.config_dir
    )
    
    # 合并命令行参数（命令行优先）
    config = merge_args_to_config(config, args)
    
    # 设置 GPU
    gpu_id = config.get('gpu_id', '0')
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # 设置随机种子
    seed = config.get('seed', 2077)
    set_seed(seed)
    
    # 获取 Trainer
    method_name = config.get('method', {}).get('name', args.method)
    if method_name == 'dca_clip':
        from core.dca_clip import DCAClipTrainer
        trainer = DCAClipTrainer(config)
    elif method_name == 'dca_coop':
        from core.dca_coop import DCACoOpTrainer
        trainer = DCACoOpTrainer(config)
    elif method_name == 'multi_prompt':
        from core.multi_prompt import MultiPromptTrainer
        trainer = MultiPromptTrainer(config)
    elif method_name == 'dca_coop_multi':
        from core.dca_coop_multi import DCACoOpMultiTrainer
        trainer = DCACoOpMultiTrainer(config)
    else:
        from core.dca import DCATrainer
        trainer = DCATrainer(config)
    
    # 打印配置信息
    source_idx = args.source
    target_idx = args.target
    mode = args.mode
    
    print("=" * 50)
    print(f"方法: {method_name}")
    print(f"数据集: {args.dataset}")
    print(f"源域: {trainer.domains[source_idx]} (idx={source_idx})")
    if target_idx is not None:
        print(f"目标域: {trainer.domains[target_idx]} (idx={target_idx})")
    print(f"模式: {mode}")
    print(f"设备: {config.get('device', 'cuda')}")
    print(f"批大小: {config.get('batch_size', 64)}")
    print("=" * 50)
    
    # 执行训练
    if mode == 'source' or mode == 'all':
        print("\n[阶段 1] 训练源域模型...")
        trainer.train_source(source_idx)
    
    if (mode == 'target' or mode == 'all') and target_idx is not None:
        print("\n[阶段 2] 目标域适应...")
        trainer.train_target(source_idx, target_idx)
    
    print("\n完成!")


if __name__ == "__main__":
    main()
