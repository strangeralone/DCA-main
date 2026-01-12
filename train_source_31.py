# 源域模型训练脚本（31 类设置）  # 标识该文件专注于 Office-31 源域训练
# 对应论文《Dual Classifier Adaptation: Source-Free UDA via Adaptive Pseudo-labels Learning》  # 说明实现出处
# 负责在 Office-31 数据集上完成源域模型训练与评估阶段  # 概述脚本用途
# -------------------------------------------------------------  # 逻辑分隔线
import argparse  # 解析命令行参数
import os  # 访问操作系统功能
import sys  # 预留系统相关操作
import os.path as osp  # 使用 osp 简写进行路径拼接
import torchvision  # 提供数据增强与模型工具
import numpy as np  # 数值运算库
import torch  # 深度学习框架核心库
import torch.nn as nn  # 神经网络层定义模块
import torch.optim as optim  # 优化器模块
from torchvision import transforms  # 导入图像变换工具
import network_new  # 引入自定义网络结构文件
from network_new import *  # 直接使用网络模块中的全部成员
from torch.utils.data import DataLoader  # 构建批量数据加载器
from data_list import ImageList, ImageList_idx  # 导入自定义数据集包装类
import random  # Python 随机数库
import pdb  # 预留断点调试功能
import math  # 提供数学函数
import copy  # 提供深拷贝工具
from loss import *  # 导入损失函数实现
import torch.nn.functional as F  # 函数式 API
from utils import *  # 工具函数集合
# -------------------------------------------------------------  # 逻辑分隔线
def op_copy(optimizer):  # 为优化器记录初始学习率
    for param_group in optimizer.param_groups:  # 遍历每个参数组
        param_group['lr0'] = param_group['lr']  # 记录最初学习率数值
    return optimizer  # 返回带有 lr0 的优化器
# -------------------------------------------------------------  # 逻辑分隔线
def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):  # 多项式学习率调度
    decay = (1 + gamma * iter_num / max_iter) ** (-power)  # 依据论文公式计算衰减因子
    for param_group in optimizer.param_groups:  # 对每个参数组应用衰减
        param_group['lr'] = param_group['lr0'] * decay  # 更新当前学习率
        param_group['weight_decay'] = 5e-4  # 设置权重衰减
        param_group['momentum'] = 0.9  # 设置动量参数
        param_group['nesterov'] = True  # 启用 Nesterov 动量
    return optimizer  # 返回调整后的优化器
# -------------------------------------------------------------  # 逻辑分隔线
def data_load(args):  # 构建数据加载器
    train_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 定义训练阶段数据增强
    test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 定义测试阶段预处理
    dsets = {}  # 保存数据集对象
    dset_loaders = {}  # 保存数据加载器
    train_bs = args.batch_size  # 读取批大小配置
    txt_src = open(args.s_dset_path).readlines()  # 读取源域文件列表
    txt_tar = open(args.t_dset_path).readlines()  # 读取目标域文件列表
    txt_test = open(args.test_dset_path).readlines()  # 读取测试文件列表
    if args.trte == "val":  # 根据设置划分验证集
        dsize = len(txt_src)  # 源域样本总量
        tr_size = int(args.split * dsize)  # 训练集数量
        test_size = dsize - tr_size  # 验证集数量
        print(dsize, tr_size, test_size)  # 输出划分信息
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, test_size])  # 随机划分训练与验证
    else:  # 不划分时的处理
        tr_txt = txt_src  # 使用所有源域数据训练
        te_txt = txt_src  # 验证同样使用全部样本
    dsets["source_tr"] = ImageList(tr_txt, transform=train_transform)  # 构建源域训练数据集
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)  # 创建源域训练加载器
    dsets["source_te"] = ImageList(te_txt, transform=test_transform)  # 构建源域验证数据集
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)  # 创建源域验证加载器
    dsets["test"] = ImageList(txt_test, transform=test_transform)  # 构建目标测试数据集
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker, drop_last=False)  # 创建测试加载器
    return dset_loaders  # 返回全部加载器
# -------------------------------------------------------------  # 逻辑分隔线
def train_source(args):  # 源域训练主流程
    dset_loaders = data_load(args)  # 构建数据加载器
    netG = network_new.ResBase(res_name=args.net).cuda()  # 初始化特征提取器并放入 GPU
    netF = network_new.bottleneck(type=args.classifier, feature_dim=netG.in_features, bottleneck_dim=args.bottleneck).cuda()  # 初始化瓶颈层
    netC = network_new.classifier_C(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()  # 初始化源分类器
    netD = network_new.classifier_D(type=args.layer, feature_dim=netG.in_features, class_num=args.class_num).cuda()  # 初始化辅助分类器
    param_group_g = []  # 特征与瓶颈层参数列表
    param_group_c = []  # 源分类器参数列表
    param_group_d = []  # 辅助分类器参数列表
    learning_rate = args.lr  # 读取学习率
    for _, v in netG.named_parameters():  # 遍历特征提取器参数
        param_group_g.append({"params": v, "lr": learning_rate * 0.1})  # 设置较小学习率
    for _, v in netF.named_parameters():  # 遍历瓶颈层参数
        param_group_g.append({"params": v, "lr": learning_rate * 1.0})  # 使用基准学习率
    for _, v in netC.named_parameters():
        param_group_c.append({"params": v, "lr": learning_rate * 1.0})
    for _, v in netD.named_parameters():  # 遍历辅助分类器参数
        param_group_d.append({"params": v, "lr": learning_rate * 1.0})  # 使用基准学习率
    optimizer_g = optim.SGD(param_group_g)  # 构建特征优化器
    optimizer_c = optim.SGD(param_group_c)  # 构建源分类器优化器
    optimizer_d = optim.SGD(param_group_d)  # 构建辅助分类器优化器
    optimizer_g = op_copy(optimizer_g)  # 记录初始 lr
    optimizer_c = op_copy(optimizer_c)  # 记录初始 lr
    optimizer_d = op_copy(optimizer_d)  # 记录初始 lr
    netG.train()  # 将特征提取器置于训练模式
    netF.train()  # 将瓶颈层置于训练模式
    netC.train()  # 将源分类器置于训练模式
    netD.train()  # 将辅助分类器置于训练模式
    acc_init = 0  # 初始化最佳精度
    iter_num = 0  # 初始化迭代计数
    iter_source = iter(dset_loaders["source_tr"])  # 构建训练数据迭代器
    interval_iter = int(args.interval * len(dset_loaders["source_tr"]))  # 计算评估间隔
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])  # 计算最大迭代次数
    while iter_num < max_iter:  # 循环训练直至达到最大迭代
        try:  # 尝试读取下一个批次
            inputs_source, labels_source = next(iter_source)  # 获取图像与标签
        except StopIteration:  # 如果迭代器耗尽
            iter_source = iter(dset_loaders["source_tr"])  # 重新创建迭代器
            inputs_source, labels_source = next(iter_source)  # 再次获取批次
        if inputs_source.size(0) == 1:  # 跳过批量为 1 的情况
            continue  # 进入下一循环
        iter_num += 1  # 递增迭代计数
        lr_scheduler(optimizer_g, iter_num=iter_num, max_iter=max_iter)  # 更新特征优化器学习率
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)  # 更新源分类器学习率
        lr_scheduler(optimizer_d, iter_num=iter_num, max_iter=max_iter)  # 更新辅助分类器学习率
        inputs_source = inputs_source.cuda()  # 将图像移动到 GPU
        labels_source = labels_source.cuda()  # 将标签移动到 GPU
        features_d = netG(inputs_source)  # 经过特征提取器获得特征
        features = netF(features_d)  # 经过瓶颈层获得投影特征
        outputs_source1 = netC(features)  # 源分类器预测
        outputs_source2 = netD(features_d)  # 辅助分类器预测
        classifier_loss1 = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source1, labels_source)  # 计算主分类器损失
        classifier_loss2 = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source2, labels_source)  # 计算辅助分类器损失
        classifier_loss = classifier_loss1 + classifier_loss2  # 合并损失
        all_loss = classifier_loss  # 设置最终损失
        optimizer_g.zero_grad()  # 清空梯度
        optimizer_c.zero_grad()  # 清空梯度
        optimizer_d.zero_grad()  # 清空梯度
        all_loss.backward()  # 反向传播
        optimizer_g.step()  # 更新特征提取器参数
        optimizer_c.step()  # 更新源分类器参数
        optimizer_d.step()  # 更新辅助分类器参数
        if iter_num % interval_iter == 0 or iter_num == max_iter:  # 到达评估节点
            netG.eval()  # 切换到评估模式
            netF.eval()  # 切换到评估模式
            netC.eval()  # 切换到评估模式
            netD.eval()  # 切换到评估模式
            _, acc_list1, accuracy1 = cal_acc(dset_loaders["source_te"], netG, netF, netC)  # 评估主分类器精度
            _, acc_list2, accuracy2 = cal_acc_easy(dset_loaders["source_te"], netG, netD)  # 评估辅助分类器精度
            acc_best = accuracy1  # 使用主分类器精度作为记录
            log_str = "Task: {}, Iter:{}; Accuracy_c = {:.2f}%, Accuracy_d = {:.2f}%\\n{}".format(args.name_src, iter_num, accuracy1, accuracy2, acc_list1)  # 构建日志文本
            args.out_file.write(log_str + "\\n")  # 写入日志文件
            args.out_file.flush()  # 刷新输出
            print(log_str + "\\n")  # 控制台打印日志
            if acc_best >= acc_init:  # 如果精度刷新最佳
                acc_init = acc_best  # 更新最佳记录
                torch.save(netG.state_dict(), osp.join(args.output_dir_src, "source_G.pt"))  # 保存特征提取器
                torch.save(netF.state_dict(), osp.join(args.output_dir_src, "source_F.pt"))  # 保存瓶颈层
                torch.save(netC.state_dict(), osp.join(args.output_dir_src, "source_C.pt"))  # 保存主分类器
                torch.save(netD.state_dict(), osp.join(args.output_dir_src, "source_D.pt"))  # 保存辅助分类器
            netG.train()  # 恢复训练模式
            netF.train()  # 恢复训练模式
            netC.train()  # 恢复训练模式
            netD.train()  # 恢复训练模式
    print('Best Model Saved!!')  # 输出最佳模型保存提示
    return netG, netF, netC, netD  # 返回训练后的网络
# -------------------------------------------------------------  # 逻辑分隔线
def test_target(args):  # 测试已训练模型
    dset_loaders = data_load(args)  # 构建评估数据加载器
    netG = network_new.ResBase(res_name=args.net).cuda()  # 初始化特征提取器
    netF = network_new.bottleneck(type=args.classifier, feature_dim=netG.in_features, bottleneck_dim=args.bottleneck).cuda()  # 初始化瓶颈层
    netC = network_new.classifier_C(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()  # 初始化主分类器
    netD = network_new.classifier_D(type=args.layer, feature_dim=netG.in_features, class_num=args.class_num).cuda()  # 初始化辅助分类器
    netG.load_state_dict(torch.load(args.output_dir_src + "/source_G.pt"))  # 加载特征提取器权重
    netF.load_state_dict(torch.load(args.output_dir_src + "/source_F.pt"))  # 加载瓶颈层权重
    netC.load_state_dict(torch.load(args.output_dir_src + "/source_C.pt"))  # 加载主分类器权重
    netD.load_state_dict(torch.load(args.output_dir_src + "/source_D.pt"))  # 加载辅助分类器权重
    netG.eval()  # 设置评估模式
    netF.eval()  # 设置评估模式
    netC.eval()  # 设置评估模式
    netD.eval()  # 设置评估模式
    acc1, acc_list1, accuracy1 = cal_acc(dset_loaders["test"], netG, netF, netC)  # 评估主分类器
    acc2, acc_list2, accuracy2 = cal_acc_easy(dset_loaders["test"], netG, netD)  # 评估辅助分类器
    log_str = "\\nDateset: {}, Task: {}, Accuracy_c = {:.2f}%, Accuracy_d = {:.2f}%\\n{}".format(args.dset, args.name, accuracy1, accuracy2, acc_list1)  # 组装日志
    args.out_file.write(log_str + "\\n")  # 写入日志
    args.out_file.flush()  # 刷新输出
    print(log_str + "\\n")  # 控制台打印
# -------------------------------------------------------------  # 逻辑分隔线
def print_args(args):  # 将参数字典格式化为字符串
    s = "==========================================\\n"  # 初始化分隔线
    for arg, content in args.__dict__.items():  # 遍历参数
        s += "{}:{}\\n".format(arg, content)  # 追加每一项配置
    return s  # 返回字符串
# -------------------------------------------------------------  # 逻辑分隔线
if __name__ == "__main__":  # 主程序入口
    parser = argparse.ArgumentParser(description="DCA on office-31")  # 创建解析器
    parser.add_argument("--gpu_id", type=str, nargs="?", default="0", help="device id to run")  # 指定 GPU 编号
    parser.add_argument("--s", type=int, default=0, help="source")  # 源域索引
    parser.add_argument("--t", type=int, default=0, help="target")  # 目标域索引
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")  # 批大小
    parser.add_argument("--worker", type=int, default=4, help="number of workers")  # 数据加载线程数
    parser.add_argument("--dset", type=str, default="office", choices=["office", "officehome", "visda"])  # 数据集选择
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")  # 学习率
    parser.add_argument("--net", type=str, default="resnet50", help="resnet50, resnet101")  # 骨干网络
    parser.add_argument("--seed", type=int, default=2042, help="random seed")  # 随机种子
    parser.add_argument("--max_epoch", type=int, default=20, help="max iterations")  # 最大轮数
    parser.add_argument("--interval", type=float, default=0.2, help="max iterations")  # 评估间隔比例
    parser.add_argument("--bottleneck", type=int, default=256)  # 瓶颈维度
    parser.add_argument("--epsilon", type=float, default=1e-5)  # 数值稳定项
    parser.add_argument("--layer", type=str, default="dca", choices=["linear", "wn", "mme", "dca"])  # 分类器类型
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])  # 瓶颈结构
    parser.add_argument("--smooth", type=float, default=0.1)  # 标签平滑系数
    parser.add_argument("--output", type=str, default="double")  # 输出配置
    parser.add_argument("--da", type=str, default="SFDA")  # 域适应标记
    parser.add_argument("--trte", type=str, default="val", choices=["full", "val"])  # 训练测试拆分策略
    parser.add_argument("--split", type=float, default=0.9, help="split parameter")  # 训练集占比
    args = parser.parse_args()  # 解析参数
    args.interval = args.max_epoch / 10  # 将评估间隔设置为总 epoch 的十分之一
    if args.dset == "office":  # Office 数据集配置
        names = ["amazon", "dslr", "webcam"]  # 域名称列表
        args.class_num = 31  # 类别数量
    elif args.dset == "officehome":  # Office-Home 数据集配置
        names = ["Art", "Clipart", "Product", "RealWorld"]  # 域名称列表
        args.class_num = 65  # 类别数量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # 环境变量指定 GPU
    SEED = args.seed  # 读取种子值
    torch.manual_seed(SEED)  # 设置 PyTorch CPU 随机种子
    torch.cuda.manual_seed(SEED)  # 设置 PyTorch GPU 随机种子
    np.random.seed(SEED)  # 设置 numpy 随机种子
    random.seed(SEED)  # 设置 Python 随机种子o
    folder = "./data/"  # 数据根目录
    if args.dset == "office":  # Office 数据集路径构建
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_31.txt"  # 源域列表
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_31.txt"  # 目标域列表
    elif args.dset == "officehome":  # Office-Home 数据集路径构建
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_65.txt"  # 源域列表
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_65.txt"  # 目标域列表
    args.test_dset_path = args.t_dset_path  # 测试集路径与目标域保持一致
    current_folder = "./ckps/"  # 模型保存根目录
    args.output_dir_src = osp.join(current_folder, args.da, args.output, args.dset, names[args.s][0].upper())  # 源模型输出目录
    args.name_src = names[args.s][0].upper()  # 源域代号
    if not osp.exists(args.output_dir_src):  # 若目录不存在
        os.system("mkdir -p " + args.output_dir_src)  # 创建目录
    if not osp.exists(args.output_dir_src):  # 若仍不存在
        os.mkdir(args.output_dir_src)  # 再次创建
    args.output_dir = osp.join(current_folder, args.da, args.output, args.dset, names[args.s][0].upper() + names[args.t][0].upper())  # 源到目标输出目录
    args.name = names[args.s][0].upper() + names[args.t][0].upper()  # 任务名称
    if not osp.exists(args.output_dir):  # 目录不存在则创建
        os.system("mkdir -p " + args.output_dir)  # 创建目录
    if not osp.exists(args.output_dir):  # 再次检查
        os.mkdir(args.output_dir)  # 再次创建
    args.out_file = open(osp.join(args.output_dir_src, "log.txt"), "w")  # 打开训练日志文件
    args.out_file.write(print_args(args) + "\n")  # 写入配置
    args.out_file.flush()  # 刷新文件
    train_source(args)  # 执行源域训练
    test_target(args)  # 对初始目标域评估
    args.out_file = open(osp.join(args.output_dir_src, 'model_test.txt'), 'w')  # 打开跨域测试日志
    for i in range(len(names)):  # 遍历全部域
        if i == args.s:  # 跳过源域自身
            continue  # 进入下一域
        args.t = i  # 更新目标域索引
        args.name = names[args.s][0].upper() + names[args.t][0].upper()  # 更新任务代号
        folder = "./data/"  # 重申数据根目录
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_31.txt"  # 更新源域路径
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_31.txt"  # 更新目标域路径
        args.test_dset_path = args.t_dset_path  # 测试路径同步
        test_target(args)  # 对新目标域进行评估
