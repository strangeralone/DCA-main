"""DCA 项目配置加载工具

提供 YAML 配置文件加载、合并和命令行参数处理功能。
支持从 conf/ 目录加载默认配置、数据集配置和方法配置。

Usage:
    from config_utils import load_config, get_arg_parser
    
    config = load_config(method='dca', dataset='officehome')
"""

import os
import argparse
import yaml
from typing import Dict, Any


def load_yaml(path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件
    
    Args:
        path: 配置文件路径
    
    Returns:
        配置字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并两个字典
    
    Args:
        base: 基础字典
        override: 覆盖字典
    
    Returns:
        合并后的字典（override 优先）
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    method: str = 'dca',
    dataset: str = 'officehome',
    config_dir: str = './conf'
) -> Dict[str, Any]:
    """加载完整配置
    
    合并顺序：default.yaml -> dataset/*.yaml -> method/*.yaml
    
    Args:
        method: 方法名称 ('dca', 'dca_clip')
        dataset: 数据集名称 ('officehome', 'office')
        config_dir: 配置文件目录
    
    Returns:
        合并后的配置字典，包含 'dataset' 和 'method' 两个子字典
    """
    # 加载默认配置
    default_path = os.path.join(config_dir, 'default.yaml')
    config = load_yaml(default_path) if os.path.exists(default_path) else {}
    
    # 加载数据集配置
    dataset_path = os.path.join(config_dir, 'dataset', f'{dataset}.yaml')
    if os.path.exists(dataset_path):
        dataset_config = load_yaml(dataset_path)
        config['dataset'] = dataset_config
    
    # 加载方法配置
    method_path = os.path.join(config_dir, 'method', f'{method}.yaml')
    if os.path.exists(method_path):
        method_config = load_yaml(method_path)
        config['method'] = method_config
    
    return config


def get_arg_parser() -> argparse.ArgumentParser:
    """获取命令行参数解析器
    
    Returns:
        配置好的 ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(
        description="DCA - Dual Classifier Adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --method dca --source 0 --mode source
  python main.py --method dca_clip --source 0 --target 1 --mode target
  python main.py --method dca --source 0 --target 1 --mode all
        """
    )
    
    # 任务参数
    parser.add_argument('--method', type=str, default='dca',
                        choices=['dca', 'dca_clip', 'dca_coop', 'multi_prompt', 'dca_coop_multi'],
                        help='训练方法: dca, dca_clip, dca_coop, multi_prompt, dca_coop_multi')
    parser.add_argument('--dataset', type=str, default='officehome',
                        choices=['officehome', 'office'],
                        help='数据集名称')
    parser.add_argument('--source', '-s', type=int, default=0,
                        help='源域索引 (0-3 for officehome, 0-2 for office)')
    parser.add_argument('--target', '-t', type=int, default=None,
                        help='目标域索引，None 表示仅训练源域')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['source', 'target', 'all'],
                        help='训练模式: source/target/all')
    
    # 覆盖配置（命令行参数优先于配置文件）
    parser.add_argument('--gpu_id', type=str, default=None,
                        help='GPU ID (覆盖配置文件)')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备 (cuda/mps/cpu)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批大小')
    parser.add_argument('--max_epoch', type=int, default=None,
                        help='最大训练轮数')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    
    # 配置目录
    parser.add_argument('--config_dir', type=str, default='./conf',
                        help='配置文件目录')
    
    return parser


def merge_args_to_config(config: Dict, args: argparse.Namespace) -> Dict:
    """将命令行参数合并到配置中
    
    命令行参数优先于配置文件中的值。
    
    Args:
        config: 配置字典
        args: 命令行参数
    
    Returns:
        更新后的配置字典
    """
    # 覆盖顶级配置
    if args.gpu_id is not None:
        config['gpu_id'] = args.gpu_id
    if args.device is not None:
        config['device'] = args.device
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.seed is not None:
        config['seed'] = args.seed
    
    # 覆盖 epoch
    if args.max_epoch is not None:
        if 'method' in config:
            if args.mode in ['source', 'all']:
                if 'source' not in config['method']:
                    config['method']['source'] = {}
                config['method']['source']['max_epoch'] = args.max_epoch
            if args.mode in ['target', 'all']:
                if 'target' not in config['method']:
                    config['method']['target'] = {}
                config['method']['target']['max_epoch'] = args.max_epoch
    
    return config
