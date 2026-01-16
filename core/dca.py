"""DCA 训练器模块

实现纯 DCA (Dual Classifier Adaptation) 的源域训练和目标域适应逻辑。
论文：Dual Classifier Adaptation: Source-Free UDA via Adaptive Pseudo-labels Learning
"""

import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import random
from datetime import datetime
from tqdm import tqdm

import utils.network_new as network_new
from utils.data_list import ImageList, ImageList_idx
from utils.loss import CrossEntropyLabelSmooth, SKL, entropy, adentropy, class_balance
from utils.helpers import cal_acc, cal_acc_easy, obtain_label, obtain_label_easy


def op_copy(optimizer):
    """保存每个参数组的初始学习率，用于学习率调度。"""
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    """多项式学习率衰减策略。"""
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 5e-4
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


class DCATrainer:
    """DCA 训练器类
    
    封装 Dual Classifier Adaptation 的完整训练流程，
    包括源域预训练和目标域无源自适应。
    """
    
    def __init__(self, config):
        """初始化训练器"""
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # 数据集配置
        self.dataset_config = config.get('dataset', {})
        self.class_num = self.dataset_config.get('class_num', 65)
        self.domains = self.dataset_config.get('domains', [])
        
        # 方法配置
        self.method_config = config.get('method', {})
        self.method_name = self.method_config.get('name', 'dca')
        
        # 网络实例
        self.netG = None
        self.netF = None
        self.netC = None
        self.netD = None
    
    def _write_config_to_log(self, log_file, task_type, source_idx, target_idx=None):
        """将配置信息写入日志文件
        
        Args:
            log_file: 日志文件句柄
            task_type: 任务类型 ('source' 或 'target')
            source_idx: 源域索引
            target_idx: 目标域索引
        """
        log_file.write("=" * 60 + "\n")
        log_file.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 60 + "\n")
        log_file.write(f"任务类型: {task_type}\n")
        log_file.write(f"方法: {self.method_name}\n")
        log_file.write(f"源域: {self.domains[source_idx]} (idx={source_idx})\n")
        if target_idx is not None:
            log_file.write(f"目标域: {self.domains[target_idx]} (idx={target_idx})\n")
        log_file.write("-" * 60 + "\n")
        log_file.write("配置信息:\n")
        
        # 写入所有配置
        log_file.write(f"  device: {self.config.get('device')}\n")
        log_file.write(f"  seed: {self.config.get('seed')}\n")
        log_file.write(f"  batch_size: {self.config.get('batch_size')}\n")
        log_file.write(f"  lr: {self.config.get('lr')}\n")
        log_file.write(f"  net: {self.config.get('net')}\n")
        log_file.write(f"  bottleneck: {self.config.get('bottleneck')}\n")
        log_file.write(f"  class_num: {self.class_num}\n")
        
        # 方法配置
        log_file.write(f"  lamda: {self.method_config.get('lamda')}\n")
        log_file.write(f"  cls_par: {self.method_config.get('cls_par')}\n")
        log_file.write(f"  alpha: {self.method_config.get('alpha')}\n")
        log_file.write(f"  mix: {self.method_config.get('mix')}\n")
        
        if task_type == 'source':
            source_cfg = self.method_config.get('source', {})
            log_file.write(f"  source.max_epoch: {source_cfg.get('max_epoch')}\n")
            log_file.write(f"  source.smooth: {source_cfg.get('smooth')}\n")
            log_file.write(f"  source.split: {source_cfg.get('split')}\n")
        else:
            target_cfg = self.method_config.get('target', {})
            log_file.write(f"  target.max_epoch: {target_cfg.get('max_epoch')}\n")
            log_file.write(f"  target.interval: {target_cfg.get('interval')}\n")
        
        log_file.write("=" * 60 + "\n\n")
        log_file.flush()
    
    def _write_end_time(self, log_file, start_time):
        """写入结束时间和耗时"""
        end_time = datetime.now()
        duration = end_time - start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        log_file.write("\n" + "=" * 60 + "\n")
        log_file.write(f"训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"总耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒\n")
        log_file.write("=" * 60 + "\n")
        log_file.flush()
        
    def _build_networks(self):
        """构建双分类器网络架构"""
        net_name = self.config.get('net', 'resnet50')
        bottleneck_dim = self.config.get('bottleneck', 256)
        classifier_type = self.config.get('classifier', 'bn')
        layer_type = self.config.get('layer', 'dca')
        
        self.netG = network_new.ResBase(res_name=net_name).to(self.device)
        self.netF = network_new.bottleneck(
            type=classifier_type, 
            feature_dim=self.netG.in_features, 
            bottleneck_dim=bottleneck_dim
        ).to(self.device)
        self.netC = network_new.classifier_C(
            type=layer_type, 
            class_num=self.class_num, 
            bottleneck_dim=bottleneck_dim
        ).to(self.device)
        self.netD = network_new.classifier_D(
            type=layer_type, 
            feature_dim=self.netG.in_features, 
            class_num=self.class_num
        ).to(self.device)
        
    def _get_transforms(self):
        """获取数据增强变换"""
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return train_transform, test_transform
    
    def _get_output_dir(self, source_idx, target_idx=None):
        """获取输出目录路径"""
        output_base = self.config.get('output_dir', './ckps')
        dataset_name = self.dataset_config.get('name', 'officehome')
        
        if target_idx is None:
            domain_name = self.domains[source_idx][0].upper()
            return osp.join(output_base, dataset_name, 'source', domain_name)
        else:
            method_name = self.method_name
            task_name = self.domains[source_idx][0].upper() + self.domains[target_idx][0].upper()
            return osp.join(output_base, dataset_name, method_name, task_name)
    
    def _load_source_data(self, source_idx):
        """加载源域数据"""
        train_transform, test_transform = self._get_transforms()
        batch_size = self.config.get('batch_size', 64)
        worker = self.config.get('worker', 4)
        
        data_dir = self.dataset_config.get('data_dir', './data/officehome')
        suffix = self.dataset_config.get('suffix', '_65.txt')
        domain_name = self.domains[source_idx]
        
        s_dset_path = osp.join(data_dir, f"{domain_name}{suffix}")
        txt_src = open(s_dset_path).readlines()
        
        source_config = self.method_config.get('source', {})
        split_ratio = source_config.get('split', 0.9)
        trte = source_config.get('trte', 'val')
        
        dsets = {}
        dset_loaders = {}
        
        if trte == "val":
            dsize = len(txt_src)
            tr_size = int(split_ratio * dsize)
            test_size = dsize - tr_size
            tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, test_size])
        else:
            tr_txt = txt_src
            te_txt = txt_src
        
        dsets["source_tr"] = ImageList(tr_txt, transform=train_transform)
        dset_loaders["source_tr"] = DataLoader(
            dsets["source_tr"], batch_size=batch_size, shuffle=True,
            num_workers=worker, drop_last=False,
        )
        dsets["source_te"] = ImageList(te_txt, transform=test_transform)
        dset_loaders["source_te"] = DataLoader(
            dsets["source_te"], batch_size=batch_size, shuffle=True,
            num_workers=worker, drop_last=False,
        )
        
        return dset_loaders
    
    def _load_target_data(self, source_idx, target_idx):
        """加载目标域数据"""
        train_transform, test_transform = self._get_transforms()
        batch_size = self.config.get('batch_size', 64)
        worker = self.config.get('worker', 4)
        
        data_dir = self.dataset_config.get('data_dir', './data/officehome')
        suffix = self.dataset_config.get('suffix', '_65.txt')
        target_domain = self.domains[target_idx]
        
        t_dset_path = osp.join(data_dir, f"{target_domain}{suffix}")
        txt_tar = open(t_dset_path).readlines()
        txt_test = txt_tar
        
        dsets = {}
        dset_loaders = {}
        
        dsets["target"] = ImageList_idx(txt_tar, transform=train_transform)
        dset_loaders["target"] = DataLoader(
            dsets["target"], batch_size=batch_size, shuffle=True,
            num_workers=worker, drop_last=False,
        )
        dsets["test"] = ImageList(txt_test, transform=test_transform)
        dset_loaders["test"] = DataLoader(
            dsets["test"], batch_size=batch_size * 3, shuffle=False,
            num_workers=worker, drop_last=False,
        )
        
        return dset_loaders, len(dsets["target"])
    
    def train_source(self, source_idx, log_file=None):
        """源域预训练"""
        start_time = datetime.now()
        print(f"训练源域模型: {self.domains[source_idx]}...")
        
        # 构建网络
        self._build_networks()
        
        # 加载数据
        dset_loaders = self._load_source_data(source_idx)
        
        # 从配置获取参数
        source_config = self.method_config.get('source', {})
        max_epoch = source_config.get('max_epoch', 40)
        smooth = source_config.get('smooth', 0.1)
        interval_epochs = source_config.get('interval', 4)
        lr = self.config.get('lr', 0.01)
        
        # 输出目录
        output_dir = self._get_output_dir(source_idx)
        os.makedirs(output_dir, exist_ok=True)
        
        # 日志文件
        if log_file is None:
            log_file = open(osp.join(output_dir, "log.txt"), "w")
        
        # 写入配置信息
        self._write_config_to_log(log_file, 'source', source_idx)
        
        # 设置优化器
        param_group_g = []
        param_group_c = []
        param_group_d = []
        
        for k, v in self.netG.named_parameters():
            param_group_g += [{"params": v, "lr": lr * 0.1}]
        for k, v in self.netF.named_parameters():
            param_group_g += [{"params": v, "lr": lr * 1.0}]
        for k, v in self.netC.named_parameters():
            param_group_c += [{"params": v, "lr": lr * 2.0}]
        for k, v in self.netD.named_parameters():
            param_group_d += [{"params": v, "lr": lr * 2.0}]
        
        optimizer_g = optim.SGD(param_group_g)
        optimizer_c = optim.SGD(param_group_c)
        optimizer_d = optim.SGD(param_group_d)
        
        optimizer_g = op_copy(optimizer_g)
        optimizer_c = op_copy(optimizer_c)
        optimizer_d = op_copy(optimizer_d)
        
        self.netG.train()
        self.netF.train()
        self.netC.train()
        self.netD.train()
        
        acc_init = 0
        iter_num = 0
        
        iter_source = iter(dset_loaders["source_tr"])
        interval_iter = int(max_epoch / interval_epochs * len(dset_loaders["source_tr"]))
        max_iter = max_epoch * len(dset_loaders["source_tr"])
        
        # 使用 tqdm 进度条
        pbar = tqdm(total=max_iter, desc=f"Source Training [{self.domains[source_idx]}]", ncols=120)
        
        while iter_num < max_iter:
            try:
                inputs_source, labels_source = next(iter_source)
            except:
                iter_source = iter(dset_loaders["source_tr"])
                inputs_source, labels_source = next(iter_source)
            
            if inputs_source.size(0) == 1:
                continue
            
            iter_num += 1
            pbar.update(1)
            
            lr_scheduler(optimizer_g, iter_num=iter_num, max_iter=max_iter)
            lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)
            lr_scheduler(optimizer_d, iter_num=iter_num, max_iter=max_iter)
            
            inputs_source = inputs_source.to(self.device)
            labels_source = labels_source.to(self.device)
            
            features_d = self.netG(inputs_source)
            features = self.netF(features_d)
            outputs_source1 = self.netC(features)
            outputs_source2 = self.netD(features_d)
            
            classifier_loss1 = CrossEntropyLabelSmooth(
                num_classes=self.class_num, epsilon=smooth, device=self.device
            )(outputs_source1, labels_source)
            classifier_loss2 = CrossEntropyLabelSmooth(
                num_classes=self.class_num, epsilon=smooth, device=self.device
            )(outputs_source2, labels_source)
            
            classifier_loss = classifier_loss1 + classifier_loss2
            
            optimizer_g.zero_grad()
            optimizer_c.zero_grad()
            optimizer_d.zero_grad()
            classifier_loss.backward()
            optimizer_g.step()
            optimizer_c.step()
            optimizer_d.step()
            
            # 更新进度条信息
            pbar.set_postfix({'loss': f'{classifier_loss.item():.4f}'})
            
            # 定期评估
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                self.netG.eval()
                self.netF.eval()
                self.netC.eval()
                self.netD.eval()
                
                _, acc_list1, accuracy1 = cal_acc(
                    dset_loaders["source_te"], self.netG, self.netF, self.netC, device=self.device
                )
                _, acc_list2, accuracy2 = cal_acc_easy(
                    dset_loaders["source_te"], self.netG, self.netD, device=self.device
                )
                
                acc_best = accuracy1
                log_str = f"Iter:{iter_num}/{max_iter}; Acc_c={accuracy1:.2f}%, Acc_d={accuracy2:.2f}%"
                
                log_file.write(log_str + "\n")
                log_file.flush()
                
                # 更新进度条
                pbar.set_postfix({'Acc_c': f'{accuracy1:.2f}%', 'Acc_d': f'{accuracy2:.2f}%'})
                
                if acc_best >= acc_init:
                    acc_init = acc_best
                    torch.save(self.netG.state_dict(), osp.join(output_dir, "source_G.pt"))
                    torch.save(self.netF.state_dict(), osp.join(output_dir, "source_F.pt"))
                    torch.save(self.netC.state_dict(), osp.join(output_dir, "source_C.pt"))
                    torch.save(self.netD.state_dict(), osp.join(output_dir, "source_D.pt"))
                
                self.netG.train()
                self.netF.train()
                self.netC.train()
                self.netD.train()
        
        pbar.close()
        
        # 写入结束时间
        self._write_end_time(log_file, start_time)
        
        print(f"源域模型训练完成! 最佳准确率: {acc_init:.2f}%")
        return output_dir
    
    def train_target(self, source_idx, target_idx, log_file=None):
        """目标域无源自适应"""
        start_time = datetime.now()
        task_name = f"{self.domains[source_idx]} -> {self.domains[target_idx]}"
        print(f"目标域适应: {task_name}...")
        
        # 构建网络
        self._build_networks()
        
        # 加载源域预训练模型
        source_dir = self._get_output_dir(source_idx)
        self.netG.load_state_dict(torch.load(osp.join(source_dir, "source_G.pt")))
        self.netF.load_state_dict(torch.load(osp.join(source_dir, "source_F.pt")))
        self.netC.load_state_dict(torch.load(osp.join(source_dir, "source_C.pt")))
        self.netD.load_state_dict(torch.load(osp.join(source_dir, "source_D.pt")))
        
        # 加载目标域数据
        dset_loaders, num_samples = self._load_target_data(source_idx, target_idx)
        
        # 从配置获取参数
        target_config = self.method_config.get('target', {})
        max_epoch = target_config.get('max_epoch', 15)
        interval = target_config.get('interval', 10)
        
        lamda = self.method_config.get('lamda', 0.45)
        cls_par = self.method_config.get('cls_par', 0.15)
        alpha = self.method_config.get('alpha', 0.5)
        mix = self.method_config.get('mix', 0.5)
        lr = self.config.get('lr', 0.01)
        
        # 输出目录
        output_dir = self._get_output_dir(source_idx, target_idx)
        os.makedirs(output_dir, exist_ok=True)
        
        # 日志文件
        if log_file is None:
            log_file = open(osp.join(output_dir, "log.txt"), "w")
        
        # 写入配置信息
        self._write_config_to_log(log_file, 'target', source_idx, target_idx)
        
        # 冻结 netC
        self.netC.eval()
        self.netD.train()
        for k, v in self.netC.named_parameters():
            v.requires_grad = False
        
        # 设置优化器
        param_group_g = []
        param_group_d = []
        
        for k, v in self.netG.named_parameters():
            param_group_g += [{"params": v, "lr": lr * 0.1}]
        for k, v in self.netF.named_parameters():
            param_group_g += [{"params": v, "lr": lr * 1.0}]
        for k, v in self.netD.named_parameters():
            param_group_d += [{"params": v, "lr": lr * 2.0}]
        
        optimizer_g = optim.SGD(param_group_g)
        optimizer_d = optim.SGD(param_group_d)
        
        optimizer_g = op_copy(optimizer_g)
        optimizer_d = op_copy(optimizer_d)
        
        iter_num = 0
        iter_target = iter(dset_loaders["target"])
        max_iter = max_epoch * len(dset_loaders["target"])
        interval_iter = max_iter // interval
        
        # 使用 tqdm 进度条
        pbar = tqdm(total=max_iter, desc=f"Target Adaptation [{task_name}]", ncols=120)
        
        best_acc = 0
        
        while iter_num < max_iter:
            try:
                inputs_test, _, tar_idx = next(iter_target)
            except:
                iter_target = iter(dset_loaders["target"])
                inputs_test, _, tar_idx = next(iter_target)
            
            if inputs_test.size(0) == 1:
                continue
            
            # 定期更新伪标签
            if iter_num % interval_iter == 0 and cls_par > 0:
                self.netG.eval()
                self.netF.eval()
                mem_label1 = obtain_label(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
                mem_label2 = obtain_label_easy(dset_loaders["test"], self.netG, self.netD, device=self.device)
                mem_label1 = torch.from_numpy(mem_label1).to(self.device)
                mem_label2 = torch.from_numpy(mem_label2).to(self.device)
                self.netG.train()
                self.netF.train()
            
            inputs_test = inputs_test.to(self.device)
            iter_num += 1
            pbar.update(1)
            
            lr_scheduler(optimizer_g, iter_num=iter_num, max_iter=max_iter)
            lr_scheduler(optimizer_d, iter_num=iter_num, max_iter=max_iter)
            
            # Step A: 训练 netD
            total_loss1 = 0
            features_d = self.netG(inputs_test)
            features = self.netF(features_d)
            outputs1 = self.netC(features)
            outputs2 = self.netD(features_d)
            
            softmax_out1 = nn.Softmax(dim=1)(outputs1)
            softmax_out2 = nn.Softmax(dim=1)(outputs2)
            
            loss_skl = torch.mean(torch.sum(SKL(softmax_out1, softmax_out2), dim=1))
            total_loss1 += loss_skl * 0.1
            
            loss_ent = entropy(self.netD, features_d, lamda)
            total_loss1 += loss_ent
            
            optimizer_d.zero_grad()
            total_loss1.backward()
            optimizer_d.step()
            
            # Step B: 联合训练 netG + netD
            total_loss2 = 0
            features_d = self.netG(inputs_test)
            features = self.netF(features_d)
            outputs1 = self.netC(features)
            outputs2 = self.netD(features_d)
            
            softmax_out1 = nn.Softmax(dim=1)(outputs1)
            softmax_out2 = nn.Softmax(dim=1)(outputs2)
            
            # 伪标签分类损失
            pred1 = mem_label1[tar_idx]
            pred2 = mem_label2[tar_idx]
            classifier_loss1 = nn.CrossEntropyLoss()(outputs1, pred1)
            classifier_loss2 = nn.CrossEntropyLoss()(outputs2, pred2)
            
            # 不确定性加权（添加数值稳定性处理）
            kl_distance = nn.KLDivLoss(reduction='none')
            log_sm = nn.LogSoftmax(dim=1)
            # 避免 softmax 输出中有 0 导致 NaN
            softmax_out1_stable = torch.clamp(softmax_out1, min=1e-8)
            softmax_out2_stable = torch.clamp(softmax_out2, min=1e-8)
            variance1 = torch.sum(kl_distance(log_sm(outputs1), softmax_out2_stable), dim=1)
            variance2 = torch.sum(kl_distance(log_sm(outputs2), softmax_out1_stable), dim=1)
            exp_variance1 = torch.mean(torch.exp(-variance1))
            exp_variance2 = torch.mean(torch.exp(-variance2))
            
            loss_seg1 = classifier_loss1 * exp_variance1 + torch.mean(variance1)
            loss_seg2 = classifier_loss2 * exp_variance2 + torch.mean(variance2)
            classifier_loss = alpha * loss_seg1 + (2 - alpha) * loss_seg2
            loss_cs = cls_par * classifier_loss
            total_loss2 += loss_cs
            
            # 对抗熵损失
            loss_ent1 = adentropy(self.netC, features, lamda)
            loss_ent2 = adentropy(self.netD, features_d, lamda)
            loss_mme = loss_ent1 + loss_ent2
            total_loss2 += loss_mme
            
            # 类平衡损失
            loss_cb1 = class_balance(softmax_out1, lamda)
            loss_cb2 = class_balance(softmax_out2, lamda)
            loss_cb = loss_cb1 + loss_cb2
            total_loss2 += loss_cb
            
            # MixUp 数据增强
            if mix > 0:
                alpha_mix = 0.3
                lam = np.random.beta(alpha_mix, alpha_mix)
                index = torch.randperm(inputs_test.size()[0]).to(self.device)
                mixed_input = lam * inputs_test + (1 - lam) * inputs_test[index, :]
                mixed_softout = (lam * softmax_out1 + (1 - lam) * softmax_out2[index, :]).detach()
                
                features_mix = self.netG(mixed_input)
                outputs_mixed1 = self.netC(self.netF(features_mix))
                outputs_mixed2 = self.netD(features_mix)
                
                outputs_mixed_softmax1 = torch.nn.Softmax(dim=1)(outputs_mixed1)
                outputs_mixed_softmax2 = torch.nn.Softmax(dim=1)(outputs_mixed2)
                
                loss_mix1 = mix * nn.KLDivLoss(reduction='batchmean')(outputs_mixed_softmax1.log(), mixed_softout)
                loss_mix2 = mix * nn.KLDivLoss(reduction='batchmean')(outputs_mixed_softmax2.log(), mixed_softout)
                loss_mix = loss_mix1 + loss_mix2
                total_loss2 += loss_mix
            
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            total_loss2.backward()
            optimizer_g.step()
            optimizer_d.step()
            
            # 更新进度条
            pbar.set_postfix({'Lcls': f'{loss_cs.item():.4f}', 'Lent': f'{loss_mme.item():.4f}'})
            
            # 定期评估
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                self.netG.eval()
                self.netF.eval()
                
                _, acc_list1, accuracy1 = cal_acc(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
                _, acc_list2, accuracy2 = cal_acc_easy(dset_loaders["test"], self.netG, self.netD, device=self.device)
                
                log_str = f"Iter:{iter_num}/{max_iter}; Acc_c={accuracy1:.2f}%, Acc_d={accuracy2:.2f}%; Lcls:{loss_cs.item():.4f}; Lent:{loss_mme.item():.4f}"
                
                log_file.write(log_str + "\n")
                log_file.flush()
                
                # 更新进度条和最佳准确率
                pbar.set_postfix({'Acc_c': f'{accuracy1:.2f}%', 'Acc_d': f'{accuracy2:.2f}%'})
                best_acc = max(best_acc, accuracy1)
                
                self.netG.train()
                self.netF.train()
            
            # 保存模型
            if self.config.get('savemodel', True):
                torch.save(self.netG.state_dict(), osp.join(output_dir, "target_G.pt"))
                torch.save(self.netF.state_dict(), osp.join(output_dir, "target_F.pt"))
                torch.save(self.netC.state_dict(), osp.join(output_dir, "target_C.pt"))
                torch.save(self.netD.state_dict(), osp.join(output_dir, "target_D.pt"))
        
        pbar.close()
        
        # 写入结束时间
        self._write_end_time(log_file, start_time)
        
        print(f"目标域适应完成: {task_name}! 最佳准确率: {best_acc:.2f}%")
        return output_dir
