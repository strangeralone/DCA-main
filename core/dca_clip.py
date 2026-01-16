"""DCA + CLIP 训练器模块

继承 DCA 训练器，添加 CLIP 引导的难样本蒸馏功能。
"""

import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm

import clip
from core.dca import DCATrainer, op_copy, lr_scheduler
from utils.loss import SKL, entropy, adentropy, class_balance
from utils.helpers import cal_acc, cal_acc_easy, obtain_label, obtain_label_easy


class DCAClipTrainer(DCATrainer):
    """DCA + CLIP 训练器类"""
    
    def __init__(self, config):
        super().__init__(config)
        
        clip_config = self.method_config.get('clip', {})
        self.clip_model_name = clip_config.get('model', 'ViT-B/16')
        self.clip_loss_weight = clip_config.get('loss_weight', 0.1)
        self.clip_threshold_std = clip_config.get('threshold_std', 1.0)
        self.clip_temperature = clip_config.get('temperature', 100)
        
        self.clip_model = None
        self.text_features = None
    
    def _load_clip_model(self):
        """加载并冻结 CLIP 模型"""
        print(f"加载 CLIP 模型: {self.clip_model_name}...")
        self.clip_model, _ = clip.load(self.clip_model_name, device=self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        print("CLIP 模型已加载并冻结!")
    
    def _prepare_text_features(self, class_names):
        """预计算 CLIP 文本特征"""
        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
        text_tokens = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        
        print(f"CLIP 文本特征已准备: {self.text_features.shape}")
    
    def _load_class_names(self):
        """加载类别名称"""
        data_dir = self.dataset_config.get('data_dir', './data/officehome')
        classname_file = self.dataset_config.get('classname_file', 'classname.txt')
        
        if classname_file:
            classname_path = osp.join(data_dir, classname_file)
            if osp.exists(classname_path):
                with open(classname_path, 'r') as f:
                    return [line.strip() for line in f.readlines()]
        
        return [f"class_{i}" for i in range(self.class_num)]
    
    def _write_config_to_log(self, log_file, task_type, source_idx, target_idx=None):
        """将配置信息写入日志文件（包含 CLIP 配置）"""
        super()._write_config_to_log(log_file, task_type, source_idx, target_idx)
        
        # 追加 CLIP 配置
        log_file.write("CLIP 配置:\n")
        log_file.write(f"  clip.model: {self.clip_model_name}\n")
        log_file.write(f"  clip.loss_weight: {self.clip_loss_weight}\n")
        log_file.write(f"  clip.threshold_std: {self.clip_threshold_std}\n")
        log_file.write(f"  clip.temperature: {self.clip_temperature}\n")
        log_file.write("=" * 60 + "\n\n")
        log_file.flush()
    
    def train_target(self, source_idx, target_idx, log_file=None):
        """目标域自适应（带 CLIP 引导）"""
        start_time = datetime.now()
        task_name = f"{self.domains[source_idx]} -> {self.domains[target_idx]}"
        print(f"目标域适应 (带 CLIP 引导): {task_name}...")
        
        # 构建网络
        self._build_networks()
        
        # 加载 CLIP 模型
        self._load_clip_model()
        
        # 准备文本特征
        class_names = self._load_class_names()
        self._prepare_text_features(class_names)
        
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
        pbar = tqdm(total=max_iter, desc=f"Target+CLIP [{task_name}]", ncols=120)
        
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
            
            # Step B: 联合训练 netG + netD (带 CLIP 蒸馏)
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
            
            # CLIP 蒸馏损失
            threshold = variance1.mean() + variance1.std() * self.clip_threshold_std
            high_discrepancy_mask = variance1 > threshold
            topk_indices = torch.where(high_discrepancy_mask)[0]
            
            if len(topk_indices) > 0:
                clip_input = F.interpolate(inputs_test[topk_indices], size=(224, 224), mode='bicubic')
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(clip_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    clip_logits = image_features @ self.text_features.T
                    clip_probs = F.softmax(clip_logits * self.clip_temperature, dim=1)
                
                loss_clip = F.kl_div(log_sm(outputs1[topk_indices]), clip_probs.detach(), reduction='batchmean') + \
                            F.kl_div(log_sm(outputs2[topk_indices]), clip_probs.detach(), reduction='batchmean')
            else:
                loss_clip = torch.tensor(0.0).to(self.device)
            
            total_loss2 += self.clip_loss_weight * loss_clip
            
            # 加权分类损失
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
            pbar.set_postfix({'Lcls': f'{loss_cs.item():.3f}', 'Lclip': f'{loss_clip.item():.3f}'})
            
            # 定期评估
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                self.netG.eval()
                self.netF.eval()
                
                _, acc_list1, accuracy1 = cal_acc(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
                _, acc_list2, accuracy2 = cal_acc_easy(dset_loaders["test"], self.netG, self.netD, device=self.device)
                
                log_str = f"Iter:{iter_num}/{max_iter}; Acc_c={accuracy1:.2f}%, Acc_d={accuracy2:.2f}%; Lcls:{loss_cs.item():.4f}; Lclip:{loss_clip.item():.4f}"
                
                log_file.write(log_str + "\n")
                log_file.flush()
                
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
        
        print(f"目标域适应完成 (带 CLIP): {task_name}! 最佳准确率: {best_acc:.2f}%")
        return output_dir
