"""
数据列表与数据集封装模块。

论文《Dual Classifier Adaptation: Source-Free UDA via Adaptive Pseudo-labels Learning》
需要根据不同实验的文本清单加载图像。本模块提供若干辅助方法和
``torch.utils.data.Dataset`` 实现，方便训练脚本按需读取图像、标签以及样本索引。
"""

from __future__ import annotations

import os
import os.path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import torchvision  # noqa: F401 保留以兼容旧脚本的隐式依赖


def make_dataset(
    image_list: Sequence[Union[str, bytes]],
    labels: Optional[np.ndarray],
) -> List[Tuple[str, Union[int, np.ndarray]]]:
    """组装路径与标签的元组列表。"""

    if labels is not None:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [
                (val.split()[0], np.array([int(la) for la in val.split()[1:]]))
                for val in image_list
            ]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]

    return images


def rgb_loader(path: str) -> Image.Image:
    """以 RGB 模式读取图像。"""

    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def l_loader(path: str) -> Image.Image:
    """以灰度模式读取图像。"""

    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("L")


class ImageList(Dataset):
    """最常用的数据集实现，返回 ``(image, label)``。"""

    def __init__(
        self,
        image_list: Sequence[Union[str, bytes]],
        labels: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mode: str = "RGB",
    ) -> None:
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in provided list.")

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __getitem__(self, index: int):
        """返回单个样本及标签。"""

        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """数据集中样本数量。"""

        return len(self.imgs)


class ImageList_idx(Dataset):
    """返回 ``(image, label, index)`` 的数据集，实现伪标签迭代需要的索引。"""

    def __init__(
        self,
        image_list: Sequence[Union[str, bytes]],
        labels: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mode: str = "RGB",
    ) -> None:
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in provided list.")

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __getitem__(self, index: int):
        """返回单个样本、标签和原始索引。"""

        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self) -> int:
        """数据集中样本数量。"""

        return len(self.imgs)


class ImageValueList(Dataset):
    """带有可更新样本权重的列表，用于重要性采样或置信度过滤。"""

    def __init__(
        self,
        image_list: Sequence[Union[str, bytes]],
        labels: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable = rgb_loader,
    ) -> None:
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in provided list.")

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values: Iterable[float]) -> None:
        """更新每个样本的权重值。"""

        self.values = values

    def __getitem__(self, index: int):
        """返回样本与标签，供调用方结合 ``self.values`` 使用。"""

        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """数据集中样本数量。"""

        return len(self.imgs)


def default_loader(path: str) -> Image.Image:
    """默认读取器：以 RGB 形式加载原图。"""

    return Image.open(path).convert("RGB")


class ObjectImage_list(Dataset):
    """针对已构建好 ``[(path, label), ...]`` 列表的轻量封装。"""

    def __init__(self, data_list, transform=None, loader=default_loader):
        self.imgs = data_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index: int):
        """返回 ``(image, label)``。"""

        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        """数据集中样本数量。"""

        return len(self.imgs)


class ObjectImage_mul_list(Dataset):
    """支持多种变换输出（例如强弱增强）的封装。"""

    def __init__(self, data_list, transform=None, loader=default_loader):
        self.imgs = data_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index: int):
        """根据配置返回单个样本、标签及索引。"""

        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            if type(self.transform).__name__ == "list":
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        return img, target, index

    def __len__(self) -> int:
        """数据集中样本数量。"""

        return len(self.imgs)
