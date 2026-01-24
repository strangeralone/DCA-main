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
    """Generate pseudo labels with feature refinement (k-means style) on GPU."""
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
                all_fea = feas.float()
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float()), 0)
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    
    # 保持在 GPU 上计算准确率
    all_label = all_label.to(device)
    accuracy = torch.sum(predict.float() == all_label).item() / float(all_label.size(0))

    # === GPU 加速的 K-Means 伪标签优化 ===
    # 归一化特征 (用于余弦距离)
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1).to(device)), 1)
    all_fea = F.normalize(all_fea, p=2, dim=1)
    
    K = all_output.size(1)
    aff = all_output  # [N, K]

    for _ in range(2):
        # 计算类中心: [K, N] @ [N, D] -> [K, D]
        initc = aff.t().mm(all_fea)
        initc = initc / (1e-8 + aff.sum(dim=0)[:, None])
        
        # 筛选有效类别
        cls_count = torch.eye(K).to(device)[predict].sum(dim=0)
        labelset = torch.where(cls_count > 0)[0]
        
        # 计算余弦距离: 1 - cosine_similarity
        # [N, D] @ [K_valid, D].T -> [N, K_valid]
        feat_chunk = all_fea
        center_chunk = initc[labelset]
        
        # 矩阵乘法计算相似度
        sim_mat = feat_chunk.mm(center_chunk.t())
        # 距离 = 1 - 相似度 (argmin 距离 等价于 argmax 相似度)
        pred_idx = sim_mat.argmax(dim=1)
        
        # 映射回原始 Label ID
        predict = labelset[pred_idx]

        # 更新所属关系矩阵 aff (one-hot)
        aff = torch.eye(K).to(device)[predict]

    # 最后转回 cpu numpy 方便兼容后续接口
    predict = predict.cpu().numpy().astype('int')
    
    # 计算 refine 后的准确率
    acc = np.sum(predict == all_label.cpu().numpy()) / len(all_fea)
    
    log_str = '【obtain_label】Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str + '\n')

    return predict


def obtain_label_easy(loader, netG, netC, device="cuda"):
    """Generate pseudo labels using encoder + classifier predictions only (GPU only)."""
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
                all_fea = feas.float()
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float()), 0)
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    # all_label is on GPU
    all_label = all_label.to(device)
    
    # GPU K-Means Refinement
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1).to(device)), 1)
    all_fea = F.normalize(all_fea, p=2, dim=1)

    K = all_output.size(1)
    aff = all_output # GPU Tensor

    for _ in range(2):
        initc = aff.t().mm(all_fea)
        initc = initc / (1e-8 + aff.sum(dim=0)[:, None])
        
        cls_count = torch.eye(K).to(device)[predict].sum(dim=0)
        labelset = torch.where(cls_count > 0)[0]

        feat_chunk = all_fea
        center_chunk = initc[labelset]
        
        sim_mat = feat_chunk.mm(center_chunk.t())
        pred_idx = sim_mat.argmax(dim=1)
        predict = labelset[pred_idx]

        aff = torch.eye(K).to(device)[predict]

    return predict.cpu().numpy().astype('int')


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
