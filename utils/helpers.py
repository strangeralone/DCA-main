"""常用评估、标签生成与文件工具函数集合。

配合 DCA 训练脚本完成评估、伪标签更新以及文本数据的读写。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist, cosine


def cosine_similarity(feature1, feature2):
    """Compute pairwise cosine similarity between two feature batches."""
    feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
    feature2 = F.normalize(feature2)
    similarity = feature1.mm(feature2.t())  # 计算余弦相似度
    return similarity  # 返回余弦相似度


def cal_acc(loader, netG, netF, netC, device="cuda"):
    """Evaluate accuracy using encoder + bottleneck + classifier."""
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            outputs = netC(netF(netG(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    row_sums = matrix.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(row_sums > 0, matrix.diagonal() / row_sums * 100, 0)
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    return aacc, acc, accuracy*100


def cal_acc_easy(loader, netG, netC, device="cuda"):
    """Evaluate accuracy using encoder + classifier (no bottleneck)."""
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            outputs = netC(netG(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    row_sums = matrix.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(row_sums > 0, matrix.diagonal() / row_sums * 100, 0)
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    return aacc, acc, accuracy*100


def obtain_label(loader, netG, netF, netC, device="cuda"):
    """Generate pseudo labels with feature refinement (k-means style)."""
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            feas = netF(netG(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    # the distance is cosine
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > 0)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], cosine)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = '【obtain_label】Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str + '\n')

    return predict.astype('int')


def obtain_label_easy(loader, netG, netC, device="cuda"):
    """Generate pseudo labels using encoder + classifier predictions only."""
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            feas = netG(inputs)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    max_prob, predict = torch.max(all_output, 1)

    # the distance is cosine
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > 0)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], cosine)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    return predict.astype('int')


IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    """Check whether a filename ends with a supported image suffix."""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root, label):
    """Read a label list file and expand to (path, label) tuples."""
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(" ")
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images


def list2txt(list, name):
    """save the list to txt file"""
    file = name
    if os.path.exists(file):
        os.remove(file)
    for (path, label) in list:
        with open(file, 'a+') as f:
            f.write(path+' ' + str(label)+'\n')

def get_matrix(loader, netG, netF, netC, device="cuda"):
    """Compute a confusion matrix across the evaluation split."""
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            outputs = netC(netF(netG(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float(), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                                                                 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                                                 22, 23, 24, 25, 26, 27, 28, 29, 30])
    return matrix


def label_mix(pred1, pred2):
    """Combine two pseudo-label sets by averaging disagreements."""
    same_labels = []
    for i in range(len(pred1)):
        if pred1[i] == pred1[i]:
            same_labels.append(pred1[i])
    # 获取不同索引位置的伪标签的均值
    diff_labels = []
    for i in range(len(pred1)):
        if pred1[i] != pred2[i]:
            avg_label = torch.round((pred1[i] + pred2[i]) / 2)
            diff_labels.append(avg_label.item())

            # 将相同伪标签和不同伪标签合并为新的一组伪标签
    new_labels = same_labels + diff_labels

    # 将新的一组伪标签转换为 numpy 数组
    labels_mix = np.array(new_labels)

    return labels_mix


# ==================== TTA 测试时增强 ====================

def _get_tta_transforms():
    """获取 TTA 增强变换列表
    
    返回 10 种增强组合：5 个裁剪位置 × 2 种翻转状态
    """
    from torchvision import transforms
    import torchvision
    
    normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # 五点裁剪函数
    def five_crop(img, size=224):
        """对图像进行五点裁剪：四角 + 中心"""
        w, h = img.size
        crop_h, crop_w = size, size
        
        # 计算裁剪位置
        tl = img.crop((0, 0, crop_w, crop_h))  # top-left
        tr = img.crop((w - crop_w, 0, w, crop_h))  # top-right
        bl = img.crop((0, h - crop_h, crop_w, h))  # bottom-left
        br = img.crop((w - crop_w, h - crop_h, w, h))  # bottom-right
        center = img.crop(((w - crop_w) // 2, (h - crop_h) // 2, 
                          (w + crop_w) // 2, (h + crop_h) // 2))  # center
        
        return [tl, tr, bl, br, center]
    
    return five_crop, normalize


def cal_acc_tta(loader, netG, netF, netC, device="cuda", num_augments=10):
    """使用测试时增强（TTA）计算准确率
    
    对每个样本做多次增强，取预测平均值。
    增强策略：5 个裁剪位置 × 2 种翻转状态 = 10 次预测
    
    Args:
        loader: 数据加载器（注意：需要使用原始 PIL 图像，不能预处理）
        netG: 特征提取器
        netF: 瓶颈层
        netC: 分类器
        device: 计算设备
        num_augments: 增强次数（默认 10）
    
    Returns:
        aacc: 平均每类准确率
        acc: 每类准确率字符串
        accuracy: 总体准确率（百分比）
    """
    from torchvision import transforms
    import torchvision
    
    five_crop, normalize = _get_tta_transforms()
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((256, 256))
    hflip = transforms.RandomHorizontalFlip(p=1.0)
    
    netG.eval()
    netF.eval()
    netC.eval()
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            # 获取原始图像路径和标签
            # 注意：这里假设 loader 返回的是 (image_tensor, label)
            # 我们需要对 tensor 做 TTA
            inputs = data[0]  # [B, C, H, W]
            labels = data[1]
            
            batch_size = inputs.size(0)
            batch_outputs = []
            
            for i in range(batch_size):
                img_tensor = inputs[i]  # [C, H, W]
                sample_outputs = []
                
                # 方法1：直接对 tensor 做增强（更高效）
                # 原图 + 水平翻转 = 2 种
                for flip in [False, True]:
                    if flip:
                        aug_tensor = torch.flip(img_tensor, dims=[2])  # 水平翻转
                    else:
                        aug_tensor = img_tensor
                    
                    # 五点裁剪（在 tensor 上操作）
                    _, h, w = aug_tensor.shape
                    crop_size = 224
                    
                    # 计算五个裁剪位置
                    crops = [
                        (0, 0),  # top-left
                        (w - crop_size, 0),  # top-right
                        (0, h - crop_size),  # bottom-left
                        (w - crop_size, h - crop_size),  # bottom-right
                        ((w - crop_size) // 2, (h - crop_size) // 2),  # center
                    ]
                    
                    for x, y in crops:
                        cropped = aug_tensor[:, y:y+crop_size, x:x+crop_size]
                        cropped = cropped.unsqueeze(0).to(device)  # [1, C, H, W]
                        
                        output = netC(netF(netG(cropped)))
                        output = F.softmax(output, dim=1)
                        sample_outputs.append(output)
                
                # 平均所有增强的预测
                avg_output = torch.stack(sample_outputs, dim=0).mean(dim=0)  # [1, num_classes]
                batch_outputs.append(avg_output)
            
            # 合并 batch
            batch_outputs = torch.cat(batch_outputs, dim=0)  # [B, num_classes]
            all_outputs.append(batch_outputs.cpu())
            all_labels.append(labels.float())
    
    all_output = torch.cat(all_outputs, dim=0)
    all_label = torch.cat(all_labels, dim=0)
    
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    matrix = confusion_matrix(all_label.numpy(), torch.squeeze(predict).float().numpy())
    row_sums = matrix.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(row_sums > 0, matrix.diagonal() / row_sums * 100, 0)
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    
    return aacc, acc, accuracy * 100


def cal_acc_easy_tta(loader, netG, netC, device="cuda"):
    """使用 TTA 计算准确率（无瓶颈层版本）
    
    Args:
        loader: 数据加载器
        netG: 特征提取器
        netC: 分类器
        device: 计算设备
    
    Returns:
        aacc: 平均每类准确率
        acc: 每类准确率字符串
        accuracy: 总体准确率（百分比）
    """
    netG.eval()
    netC.eval()
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            inputs = data[0]  # [B, C, H, W]
            labels = data[1]
            
            batch_size = inputs.size(0)
            batch_outputs = []
            
            for i in range(batch_size):
                img_tensor = inputs[i]  # [C, H, W]
                sample_outputs = []
                
                for flip in [False, True]:
                    if flip:
                        aug_tensor = torch.flip(img_tensor, dims=[2])
                    else:
                        aug_tensor = img_tensor
                    
                    _, h, w = aug_tensor.shape
                    crop_size = 224
                    
                    crops = [
                        (0, 0), (w - crop_size, 0), (0, h - crop_size),
                        (w - crop_size, h - crop_size), ((w - crop_size) // 2, (h - crop_size) // 2),
                    ]
                    
                    for x, y in crops:
                        cropped = aug_tensor[:, y:y+crop_size, x:x+crop_size]
                        cropped = cropped.unsqueeze(0).to(device)
                        
                        output = netC(netG(cropped))
                        output = F.softmax(output, dim=1)
                        sample_outputs.append(output)
                
                avg_output = torch.stack(sample_outputs, dim=0).mean(dim=0)
                batch_outputs.append(avg_output)
            
            batch_outputs = torch.cat(batch_outputs, dim=0)
            all_outputs.append(batch_outputs.cpu())
            all_labels.append(labels.float())
    
    all_output = torch.cat(all_outputs, dim=0)
    all_label = torch.cat(all_labels, dim=0)
    
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    matrix = confusion_matrix(all_label.numpy(), torch.squeeze(predict).float().numpy())
    row_sums = matrix.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(row_sums > 0, matrix.diagonal() / row_sums * 100, 0)
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    
    return aacc, acc, accuracy * 100
