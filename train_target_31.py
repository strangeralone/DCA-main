# 目标域自适应脚本（31 类设置）  # 表示脚本面向 Office-31 场景
# 实现论文《Dual Classifier Adaptation: Source-Free UDA via Adaptive Pseudo-labels Learning》中的目标自适应阶段  # 说明出处
# 从源模型出发在无源数据情况下适配目标域  # 描述用途
# -------------------------------------------------------------  # 分隔线
import argparse  # 解析命令行参数
import os  # 操作系统接口
import sys  # 预留系统功能
import os.path as osp  # 便捷路径操作
import torchvision  # 图像增强库
import numpy as np  # 数值运算库
import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from torchvision import transforms  # 图像变换工具
import network_new  # 自定义网络
from network_new import *  # 引入全部网络组件
from torch.utils.data import DataLoader  # 数据加载器
from data_list import ImageList, ImageList_idx  # 数据集封装
import random  # 随机数模块
import pdb  # 调试工具
import math  # 数学函数
import copy  # 拷贝工具
from loss import *  # 损失函数集合
import torch.nn.functional as F  # 函数式接口
from utils import *  # 工具函数集合
import os  # 再次导入以设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 启用同步调试模式
# -------------------------------------------------------------  # 分隔线
def op_copy(optimizer):  # 记录优化器初始学习率
    for param_group in optimizer.param_groups:  # 遍历参数组
        param_group['lr0'] = param_group['lr']  # 缓存当前学习率
    return optimizer  # 返回带 lr0 的优化器
# -------------------------------------------------------------  # 分隔线
def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):  # 多项式学习率调度
    decay = (1 + gamma * iter_num / max_iter) ** (-power)  # 计算衰减系数
    for param_group in optimizer.param_groups:  # 应用到每个参数组
        param_group['lr'] = param_group['lr0'] * decay  # 更新学习率
        param_group['weight_decay'] = 5e-4  # 设定权重衰减
        param_group['momentum'] = 0.9  # 设定动量
        param_group['nesterov'] = True  # 启用 Nesterov
    return optimizer  # 返回调整后的优化器
# -------------------------------------------------------------  # 分隔线
def data_load(args):  # 构建目标域数据加载器
    train_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 训练增强流水线
    test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 测试预处理流程
    dsets = {}  # 数据集容器
    dset_loaders = {}  # 加载器容器
    train_bs = args.batch_size  # 批大小
    txt_tar = open(args.t_dset_path).readlines()  # 读取目标域列表
    txt_test = open(args.test_dset_path).readlines()  # 读取测试列表
    dsets["target"] = ImageList_idx(txt_tar, transform=train_transform)  # 构建目标域数据集
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)  # 目标域数据加载器
    num_examp = len(dsets["target"])  # 计算样本数量
    dsets["test"] = ImageList(txt_test, transform=test_transform)  # 构建测试数据集
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker, drop_last=False)  # 测试数据加载器
    return dset_loaders, num_examp  # 返回加载器与样本数
# -------------------------------------------------------------  # 分隔线
def print_args(args):  # 格式化打印参数
    s = "==========================================\n"  # 初始分隔线
    for arg, content in args.__dict__.items():  # 遍历参数
        s += "{}:{}\n".format(arg, content)  # 追加条目
    return s  # 返回字符串
# -------------------------------------------------------------  # 分隔线
def train_target_mme(args):  # 执行目标域适应
    dset_loaders, num_examp = data_load(args)  # 加载数据及样本量
    netG = network_new.ResBase(res_name=args.net).cuda()  # 初始化特征提取器
    netF = network_new.bottleneck(type=args.classifier, feature_dim=netG.in_features, bottleneck_dim=args.bottleneck).cuda()  # 初始化瓶颈层
    netC = network_new.classifier_C(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()  # 初始化源分类器
    netD = network_new.classifier_D(type=args.layer, feature_dim=netG.in_features, class_num=args.class_num).cuda()  # 初始化目标分类器
    args.modelpath = args.output_dir_src + "/source_G.pt"  # 特征提取器权重路径
    netG.load_state_dict(torch.load(args.modelpath))  # 加载特征提取器权重
    args.modelpath = args.output_dir_src + "/source_F.pt"  # 瓶颈层权重路径
    netF.load_state_dict(torch.load(args.modelpath))  # 加载瓶颈层权重
    args.modelpath = args.output_dir_src + "/source_C.pt"  # 源分类器权重路径
    netC.load_state_dict(torch.load(args.modelpath))  # 加载源分类器权重
    args.modelpath = args.output_dir_src + "/source_D.pt"  # 目标分类器权重路径
    netD.load_state_dict(torch.load(args.modelpath))  # 加载目标分类器权重
    netC.eval()  # 冻结源分类器
    netD.train()  # 训练目标分类器
    for _, v in netC.named_parameters():  # 遍历源分类器参数
        v.requires_grad = False  # 禁止更新
    param_group_g = []  # G+F 参数集合
    param_group_d = []  # D 参数集合
    learning_rate = args.lr  # 读取学习率
    for _, v in netG.named_parameters():  # 遍历 G 参数
        param_group_g.append({"params": v, "lr": learning_rate * 0.1})  # 设置较小 LR
    for _, v in netF.named_parameters():  # 遍历 F 参数
        param_group_g.append({"params": v, "lr": learning_rate * 1.0})  # 设置基准 LR
    for _, v in netD.named_parameters():  # 遍历 D 参数
        param_group_d.append({"params": v, "lr": learning_rate * 1.0})  # 设置基准 LR
    optimizer_g = optim.SGD(param_group_g, momentum=0.9, weight_decay=5e-4, nesterov=True)  # 构建 G/F 优化器
    optimizer_d = optim.SGD(param_group_d, momentum=0.9, weight_decay=5e-4, nesterov=True)  # 构建 D 优化器
    iter_num = 0  # 初始化迭代计数
    iter_target = iter(dset_loaders["target"])  # 创建目标域迭代器
    max_iter = args.max_epoch * len(dset_loaders["target"])  # 最大迭代次数
    interval_iter = max_iter // args.interval  # 打印间隔
    while iter_num < max_iter:  # 主循环
        try:  # 尝试读取批次
            inputs_test, _, tar_idx = next(iter_target)  # 读取目标图像与索引
        except StopIteration:  # 迭代器耗尽
            iter_target = iter(dset_loaders["target"])  # 重建迭代器
            inputs_test, _, tar_idx = next(iter_target)  # 再次取批次
        if inputs_test.size(0) == 1:  # 避免 batch=1
            continue  # 直接跳过
        if iter_num % interval_iter == 0 and args.cls_par > 0:  # 定期刷新伪标签
            netG.eval()  # 暂停更新
            netF.eval()  # 同上
            mem_label1 = obtain_label(dset_loaders["test"], netG, netF, netC)  # 生成源伪标签
            mem_label2 = obtain_label_easy(dset_loaders["test"], netG, netD)  # 生成目标伪标签
            mem_label1 = torch.from_numpy(mem_label1).cuda()  # 转为 GPU 张量
            mem_label2 = torch.from_numpy(mem_label2).cuda()  # 转为 GPU 张量
            netG.train()  # 恢复训练
            netF.train()  # 恢复训练
        inputs_test = inputs_test.cuda()  # 将图像迁移到 GPU
        batch_size = inputs_test.shape[0]  # 记录批大小
        iter_num += 1  # 更新迭代次数
        total_loss1 = 0  # 初始化第一阶段损失
        features_d = netG(inputs_test)  # 提取目标特征
        features = netF(features_d)  # 通过瓶颈层
        outputs1 = netC(features)  # 源分类器预测
        outputs2 = netD(features_d)  # 目标分类器预测
        softmax_out1 = nn.Softmax(dim=1)(outputs1)  # 源预测概率
        softmax_out2 = nn.Softmax(dim=1)(outputs2)  # 目标预测概率
        loss_skl = torch.mean(torch.sum(SKL(softmax_out1, softmax_out2), dim=1))  # 计算分类器差异
        total_loss1 += loss_skl * 0.1  # 加权差异损失
        loss_ent = entropy(netD, features_d, args.lamda)  # 熵最小化损失
        total_loss1 += loss_ent  # 累积第一阶段损失
        optimizer_d.zero_grad()  # 清空 D 梯度
        total_loss1.backward()  # 反向传播
        optimizer_d.step()  # 更新 D 参数
        for _ in range(1):  # 第二阶段循环
            total_loss2 = 0  # 初始化第二阶段损失
            features_d = netG(inputs_test)  # 再次提特征
            features = netF(features_d)  # 再次降维
            outputs1 = netC(features)  # 源分类器预测
            outputs2 = netD(features_d)  # 目标分类器预测
            softmax_out1 = nn.Softmax(dim=1)(outputs1)  # 源概率
            softmax_out2 = nn.Softmax(dim=1)(outputs2)  # 目标概率
            pred1 = mem_label1[tar_idx]  # 取源伪标签
            pred2 = mem_label2[tar_idx]  # 取目标伪标签
            classifier_loss1 = nn.CrossEntropyLoss()(outputs1, pred1)  # 源交叉熵
            classifier_loss2 = nn.CrossEntropyLoss()(outputs2, pred2)  # 目标交叉熵
            kl_distance = nn.KLDivLoss(reduction='none')  # KL 损失函数
            log_sm = nn.LogSoftmax(dim=1)  # 对数 softmax
            variance1 = torch.sum(kl_distance(log_sm(outputs1), softmax_out2), dim=1)  # 源方差估计
            variance2 = torch.sum(kl_distance(log_sm(outputs2), softmax_out1), dim=1)  # 目标方差估计
            exp_variance1 = torch.mean(torch.exp(-variance1))  # 计算权重
            exp_variance2 = torch.mean(torch.exp(-variance2))  # 计算权重
            loss_seg1 = classifier_loss1 * exp_variance1 + torch.mean(variance1)  # 源伪标签损失
            loss_seg2 = classifier_loss2 * exp_variance2 + torch.mean(variance2)  # 目标伪标签损失
            classifier_loss = args.alpha * loss_seg1 + (2 - args.alpha) * loss_seg2  # 综合分类损失
            loss_cs = args.cls_par * classifier_loss  # 按参数缩放
            if iter_num < interval_iter and args.dset == "visda":  # 针对 visda 的 warmup
                loss_cs *= 0  # 置零分类损失
            total_loss2 += loss_cs  # 累加分类损失
            loss_ent1 = adentropy(netC, features, args.lamda)  # 源分类器熵损失
            loss_ent2 = adentropy(netD, features_d, args.lamda)  # 目标分类器熵损失
            loss_mme = loss_ent1 + loss_ent2  # 合并对抗熵
            total_loss2 += loss_mme  # 累加熵损失
            loss_cb1 = class_balance(softmax_out1, args.lamda)  # 源类平衡
            loss_cb2 = class_balance(softmax_out2, args.lamda)  # 目标类平衡
            loss_cb = loss_cb1 + loss_cb2  # 合并类平衡
            total_loss2 += loss_cb  # 累加类平衡
            if args.mix > 0:  # 判断是否启用 mixup
                alpha = 0.3  # mixup 超参数
                lam = np.random.beta(alpha, alpha)  # 采样 mixup 权重
                index = torch.randperm(inputs_test.size(0)).cuda()  # 随机索引
                mixed_input = lam * inputs_test + (1 - lam) * inputs_test[index, :]  # 生成混合样本
                mixed_softout = (lam * softmax_out1 + (1 - lam) * softmax_out2[index, :]).detach()  # 构造目标分布
                features_mix = netG(mixed_input)  # 对混合样本提特征
                outputs_mixed1 = netC(netF(features_mix))  # 源分类器预测混合样本
                outputs_mixed2 = netD(features_mix)  # 目标分类器预测混合样本
                outputs_mied_softmax1 = torch.nn.Softmax(dim=1)(outputs_mixed1)  # 源混合概率
                outputs_mied_softmax2 = torch.nn.Softmax(dim=1)(outputs_mixed2)  # 目标混合概率
                loss_mix1 = args.mix * nn.KLDivLoss(reduction='batchmean')(outputs_mied_softmax1.log(), mixed_softout)  # 源 mixup 损失
                loss_mix2 = args.mix * nn.KLDivLoss(reduction='batchmean')(outputs_mied_softmax2.log(), mixed_softout)  # 目标 mixup 损失
                loss_mix = loss_mix1 + loss_mix2  # 合并 mixup
            else:  # 未启用 mixup
                loss_mix = 0  # 置零
            total_loss2 += loss_mix  # 累加 mixup 损失
            optimizer_g.zero_grad()  # 清空 G/F 梯度
            optimizer_d.zero_grad()  # 清空 D 梯度
            total_loss2.backward()  # 反向传播综合损失
            optimizer_g.step()  # 更新 G/F
            optimizer_d.step()  # 更新 D
        if iter_num % interval_iter == 0 or iter_num == max_iter:  # 定期评估
            netG.eval()  # 切换评估模式
            netF.eval()  # 切换评估模式
            acc1, acc_list1, accuracy1 = cal_acc(dset_loaders["test"], netG, netF, netC)  # 评估源分类器
            acc2, acc_list2, accuracy2 = cal_acc_easy(dset_loaders["test"], netG, netD)  # 评估目标分类器
            log_str = "Task: {}, Iter:{}/{}; Accuracy_c = {:.2f}%, Accuracy_d = {:.2f}% ; Lcls : {:.6f}; Lent : {:.6f}".format(args.name, iter_num, args.max_epoch * len(dset_loaders["target"]), acc1, acc2, loss_cs.data, loss_mme.data) + "\n" + str(acc_list1)  # 组装日志
            args.out_file.write(log_str + "\n")  # 写入日志文件
            args.out_file.flush()  # 刷新输出
            print(log_str + "\n")  # 控制台输出
            netG.train()  # 恢复训练
            netF.train()  # 恢复训练
    return netG, netF, netC, netD  # 返回适应后的模型
# -------------------------------------------------------------  # 分隔线
if __name__ == "__main__":  # 程序入口
    parser = argparse.ArgumentParser(description="DCA on office-31")  # 创建解析器
    parser.add_argument("--gpu_id", type=str, nargs="?", default="0", help="device id to run")  # 指定 GPU
    parser.add_argument("--s", type=int, default=0, help="source")  # 源域索引
    parser.add_argument("--t", type=int, default=1, help="target")  # 目标域索引
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")  # 批大小
    parser.add_argument("--worker", type=int, default=4, help="number of workers")  # 数据线程
    parser.add_argument("--dset", type=str, default="office", choices=["office", "officehome", "visda"])  # 数据集选择
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")  # 学习率
    parser.add_argument("--net", type=str, default="resnet50", help="resnet50, resnet101")  # 骨干网络
    parser.add_argument("--lamda", type=float, default=0.5, metavar='LAM', help='value of lamda')  # 熵项系数
    parser.add_argument("--cls_par", type=float, default=0.3)  # 分类损失权重
    parser.add_argument("--mix", type=float, default=1.0)  # mixup 系数
    parser.add_argument("--seed", type=int, default=2077, help="random seed")  # 随机种子
    parser.add_argument("--max_epoch", type=int, default=10, help="max iterations")  # 最大轮数
    parser.add_argument("--interval", type=float, default=15, help="max iterations")  # 评估间隔分母
    parser.add_argument("--alpha", type=float, default=1.5, help="parameter1")  # 分类损失系数
    parser.add_argument("--beta", type=float, default=1.0, help="parameter2")  # 备用参数
    parser.add_argument("--epsilon", type=float, default=1e-5)  # 数值稳定项
    parser.add_argument("--bottleneck", type=int, default=256)  # 瓶颈维度
    parser.add_argument("--layer", type=str, default="dca")  # 分类器结构
    parser.add_argument("--use_amp", type=bool, default=True)  # 混合精度开关
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])  # 瓶颈类型
    parser.add_argument("--output", type=str, default="double")  # 输出配置
    parser.add_argument("--da", type=str, default="SFDA")  # 域适应模式
    args = parser.parse_args()  # 解析参数
    if args.dset == "office":  # office 配置
        names = ["amazon", "dslr", "webcam"]  # 域名称
        args.class_num = 31  # 类别数
    elif args.dset == "officehome":  # officehome 配置
        names = ["Art", "Clipart", "Product", "RealWorld"]  # 域名称
        args.class_num = 65  # 类别数
    elif args.dset == "visda":  # visda 配置
        names = ['train', 'validation']  # 域名称
        args.class_num = 12  # 类别数
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # 设置 GPU 可见性
    SEED = args.seed  # 读取随机种子
    torch.manual_seed(SEED)  # 设置 CPU 种子
    torch.cuda.manual_seed(SEED)  # 设置 GPU 种子
    np.random.seed(SEED)  # 设置 numpy 种子
    random.seed(SEED)  # 设置 Python 种子
    folder = "./data/"  # 数据目录
    if args.dset == "office":  # office 数据路径
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_31.txt"  # 源域路径
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_31.txt"  # 目标路径
    elif args.dset == "officehome":  # officehome 数据路径
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_65.txt"  # 源域路径
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_65.txt"  # 目标路径
    elif args.dset == "visda":  # visda 数据路径
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_12.txt"  # 源域路径
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_12.txt"  # 目标路径
    args.test_dset_path = args.t_dset_path  # 测试路径
    current_folder = "./ckps/"  # 模型保存根目录
    args.output_dir_src = osp.join(current_folder, args.da, args.output, args.dset, names[args.s][0].upper())  # 源模型目录
    args.name_src = names[args.s][0].upper()  # 源域简称
    if not osp.exists(args.output_dir_src):  # 若目录不存在
        os.system("mkdir -p " + args.output_dir_src)  # 创建目录
    if not osp.exists(args.output_dir_src):  # 再次确认
        os.mkdir(args.output_dir_src)  # 备用创建
    args.output_dir = osp.join(current_folder, args.da, args.output, args.dset, names[args.s][0].upper() + names[args.t][0].upper())  # 目标输出目录
    args.name = names[args.s][0].upper() + names[args.t][0].upper()  # 任务简称
    if not osp.exists(args.output_dir):  # 若目录不存在
        os.system("mkdir -p " + args.output_dir)  # 创建目录
    if not osp.exists(args.output_dir):  # 再次确认
        os.mkdir(args.output_dir)  # 备用创建
    args.out_file = open(osp.join(args.output_dir, "log_.txt"), "w")  # 打开日志文件
    args.out_file.write(print_args(args) + "\n")  # 写入参数
    args.out_file.flush()  # 刷新输出
    train_target_mme(args)  # 启动目标域适应
