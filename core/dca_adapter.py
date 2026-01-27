"""DCA + CLIP Adapter 训练器模块

继承 DCA 训练器，使用 CLIP 视觉 Adapter 适应目标域。
与 CoOp 不同，这里是在视觉端添加 Adapter，而不是学习文本提示。

核心创新：
- 双向知识迁移：DCA ↔ CLIP(Adapter)
- 用 DCA 双分类器的高置信度预测反向指导 Adapter 训练
- Adapter 让 CLIP 更好地适应目标域

参考：
- CLIP-Adapter: Better Vision-Language Models with Feature Adapters (IJCV 2023)
- DIFO: Source-Free Domain Adaptation with Frozen Multimodal Foundation Model (CVPR 2024)
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
from utils.loss import SKL, entropy, adentropy, class_balance, iid_loss
from utils.helpers import cal_acc, cal_acc_easy, obtain_label, obtain_label_easy
from utils.clip_adapter import CLIPAdapter


class DCAAdapterTrainer(DCATrainer):
    """DCA + CLIP Adapter 训练器类
    
    核心思想：
    1. 在 CLIP 视觉编码器后添加轻量 Adapter
    2. 用 DCA 双分类器的高置信度样本指导 Adapter 训练
    3. 用适应后的 CLIP(Adapter) 蒸馏 DCA 处理难样本
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Adapter 配置
        adapter_config = self.method_config.get('adapter', {})
        self.clip_model_name = adapter_config.get('model', 'ViT-B/32')
        self.adapter_bottleneck = adapter_config.get('bottleneck_ratio', 4)
        self.adapter_residual_ratio = adapter_config.get('residual_ratio', 0.2)
        self.adapter_lr = adapter_config.get('adapter_lr', 0.001)
        self.adapter_temperature = adapter_config.get('temperature', 100)
        self.adapter_loss_weight = adapter_config.get('loss_weight', 0.1)
        
        # 高置信度样本选择配置
        self.high_conf_threshold = adapter_config.get('high_conf_threshold', 0.3)
        self.consistency_threshold = adapter_config.get('consistency_threshold', 0.8)
        
        # Adapter 训练配置
        self.adapter_tuning_steps = adapter_config.get('adapter_tuning_steps', 20)
        self.warmup_epochs = adapter_config.get('warmup_epochs', 2)
        
        # 模型组件
        self.clip_model = None
        self.clip_adapter = None
        self.text_features = None  # 固定的文本特征
        
        # 调试配置
        self.debug = config.get('debug', False)
        self.verbose_loss = config.get('verbose_loss', True)
    
    def _load_clip_model(self):
        """加载 CLIP 模型"""
        print(f"加载 CLIP 模型: {self.clip_model_name}...")
        self.clip_model, _ = clip.load(self.clip_model_name, device=self.device, download_root='./vlm')
        
        # 冻结 CLIP 所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        print("CLIP 模型已加载!")
    
    def _init_clip_adapter(self):
        """初始化 CLIP 视觉 Adapter"""
        # 获取 CLIP 输出维度
        if 'ViT-B' in self.clip_model_name:
            clip_dim = 512
        elif 'ViT-L' in self.clip_model_name:
            clip_dim = 768
        else:
            clip_dim = 512  # 默认
        
        print(f"初始化 CLIP Adapter: dim={clip_dim}, bottleneck={self.adapter_bottleneck}, "
              f"residual={self.adapter_residual_ratio}")
        
        self.clip_adapter = CLIPAdapter(
            in_dim=clip_dim,
            bottleneck_ratio=self.adapter_bottleneck,
            residual_ratio=self.adapter_residual_ratio
        ).to(self.device)
        
        # 统计可训练参数
        num_params = sum(p.numel() for p in self.clip_adapter.parameters() if p.requires_grad)
        print(f"CLIP Adapter 可训练参数: {num_params:,}")
    
    def _init_text_features(self, class_names):
        """初始化固定的文本特征（不使用 Prompt Learning）"""
        print(f"初始化文本特征: {len(class_names)} 个类别")
        
        # 使用简单的提示模板
        prompts = [f"a photo of a {name.replace('_', ' ')}." for name in class_names]
        text_tokens = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        
        print(f"文本特征形状: {self.text_features.shape}")
    
    def _get_adapted_clip_features(self, images, use_adapter=True):
        """获取经过 Adapter 的 CLIP 图像特征
        
        Args:
            images: 输入图像 [batch, C, H, W]
            use_adapter: 是否使用 Adapter
            
        Returns:
            image_features: 归一化的图像特征 [batch, dim]
        """
        # 调整图像大小到 CLIP 输入尺寸
        clip_input = F.interpolate(images, size=(224, 224), mode='bicubic')
        
        # 获取 CLIP 图像特征（冻结）
        with torch.no_grad():
            image_features = self.clip_model.encode_image(clip_input)
        
        # 应用 Adapter（可学习）
        if use_adapter and self.clip_adapter is not None:
            image_features = self.clip_adapter(image_features.float())
        
        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    def _get_clip_predictions(self, images, use_adapter=True):
        """获取 CLIP 预测概率
        
        Args:
            images: 输入图像
            use_adapter: 是否使用 Adapter
            
        Returns:
            probs: 预测概率 [batch, num_classes]
        """
        image_features = self._get_adapted_clip_features(images, use_adapter)
        logits = image_features @ self.text_features.T
        probs = F.softmax(logits.float() * self.adapter_temperature, dim=1)
        return probs
    
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
        """将配置信息写入日志文件"""
        super()._write_config_to_log(log_file, task_type, source_idx, target_idx)
        
        # 追加 Adapter 配置
        log_file.write("CLIP Adapter 配置:\n")
        log_file.write(f"  adapter.model: {self.clip_model_name}\n")
        log_file.write(f"  adapter.bottleneck_ratio: {self.adapter_bottleneck}\n")
        log_file.write(f"  adapter.residual_ratio: {self.adapter_residual_ratio}\n")
        log_file.write(f"  adapter.adapter_lr: {self.adapter_lr}\n")
        log_file.write(f"  adapter.temperature: {self.adapter_temperature}\n")
        log_file.write(f"  adapter.loss_weight: {self.adapter_loss_weight}\n")
        log_file.write(f"  adapter.high_conf_threshold: {self.high_conf_threshold}\n")
        log_file.write(f"  adapter.consistency_threshold: {self.consistency_threshold}\n")
        log_file.write(f"  adapter.adapter_tuning_steps: {self.adapter_tuning_steps}\n")
        log_file.write(f"  adapter.warmup_epochs: {self.warmup_epochs}\n")
        log_file.write("=" * 60 + "\n\n")
        log_file.flush()
    
    def _collect_high_confidence_samples(self, loader):
        """收集高置信度样本（用于 Adapter 训练）
        
        使用双分类器一致性作为可靠性信号：
        - 熵低（预测确定）
        - C 和 D 预测一致（分类器达成共识）
        
        Returns:
            high_conf_indices: 高置信度样本索引
            high_conf_probs: 对应的 DCA 软标签
        """
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        self.netD.eval()
        
        all_probs_c = []
        all_probs_d = []
        all_indices = []
        
        with torch.no_grad():
            for inputs, _, idx in loader:
                inputs = inputs.to(self.device)
                
                # 获取双分类器预测
                features_d = self.netG(inputs)
                features = self.netF(features_d)
                outputs1 = self.netC(features)
                outputs2 = self.netD(features_d)
                
                softmax_out1 = F.softmax(outputs1, dim=1)
                softmax_out2 = F.softmax(outputs2, dim=1)
                
                all_probs_c.append(softmax_out1.cpu())
                all_probs_d.append(softmax_out2.cpu())
                all_indices.append(idx)
        
        all_probs_c = torch.cat(all_probs_c, dim=0)
        all_probs_d = torch.cat(all_probs_d, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        
        # 平均预测作为软标签
        avg_probs = (all_probs_c + all_probs_d) / 2
        
        # 条件 1: 低熵（高置信度）
        entropy_vals = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8), dim=1)
        max_entropy = np.log(self.class_num)
        entropy_threshold = max_entropy * self.high_conf_threshold
        low_entropy_mask = entropy_vals < entropy_threshold
        
        # 条件 2: 双分类器一致性
        pred_c = all_probs_c.argmax(dim=1)
        pred_d = all_probs_d.argmax(dim=1)
        consistency_mask = pred_c == pred_d
        
        # 条件 3: 置信度一致性（两个分类器的最大概率差异小）
        conf_c = all_probs_c.max(dim=1)[0]
        conf_d = all_probs_d.max(dim=1)[0]
        conf_diff = torch.abs(conf_c - conf_d)
        conf_consistency_mask = conf_diff < (1 - self.consistency_threshold)
        
        # 综合条件
        high_conf_mask = low_entropy_mask & consistency_mask & conf_consistency_mask
        
        high_conf_indices = all_indices[high_conf_mask]
        high_conf_probs = avg_probs[high_conf_mask]
        
        print(f"  高置信度样本: {len(high_conf_indices)}/{len(all_indices)} "
              f"({100*len(high_conf_indices)/len(all_indices):.1f}%)")
        print(f"    - 低熵: {low_entropy_mask.sum().item()}")
        print(f"    - 预测一致: {consistency_mask.sum().item()}")
        print(f"    - 置信度一致: {conf_consistency_mask.sum().item()}")
        
        return high_conf_indices, high_conf_probs
    
    def _tune_adapter(self, dset_loaders, high_conf_indices, high_conf_probs, log_file=None):
        """独立 Adapter Tuning 阶段
        
        用 DCA 的高置信度预测指导 Adapter 训练，
        让 CLIP 的输出更适应目标域。
        
        Args:
            dset_loaders: 数据加载器
            high_conf_indices: 高置信度样本索引
            high_conf_probs: DCA 软标签
            log_file: 日志文件
        """
        print(f"  开始 Adapter Tuning ({self.adapter_tuning_steps} steps)...")
        
        # 创建索引到软标签的映射
        idx_to_prob = {idx.item(): prob for idx, prob in zip(high_conf_indices, high_conf_probs)}
        
        # 设置 Adapter 优化器
        optimizer_adapter = optim.AdamW(
            self.clip_adapter.parameters(),
            lr=self.adapter_lr,
            weight_decay=0.01
        )
        
        # 冻结 DCA 网络
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        self.netD.eval()
        
        # 训练 Adapter
        self.clip_adapter.train()
        
        step = 0
        iter_loader = iter(dset_loaders["target"])
        total_loss = 0.0
        
        while step < self.adapter_tuning_steps:
            try:
                inputs, _, tar_idx = next(iter_loader)
            except StopIteration:
                iter_loader = iter(dset_loaders["target"])
                inputs, _, tar_idx = next(iter_loader)
            
            # 筛选高置信度样本
            batch_mask = torch.tensor([i.item() in idx_to_prob for i in tar_idx])
            if not batch_mask.any():
                continue
            
            inputs_high_conf = inputs[batch_mask].to(self.device)
            selected_idx = tar_idx[batch_mask]
            
            # 获取 DCA 软标签
            dca_soft_labels = torch.stack([idx_to_prob[i.item()] for i in selected_idx]).to(self.device)
            
            # 获取 Adapter 后的 CLIP 预测
            clip_probs = self._get_clip_predictions(inputs_high_conf, use_adapter=True)
            
            # IID Loss: 让 CLIP(Adapter) 输出对齐 DCA 软标签
            loss = iid_loss(clip_probs, dca_soft_labels)
            
            optimizer_adapter.zero_grad()
            loss.backward()
            optimizer_adapter.step()
            
            total_loss += loss.item()
            step += 1
        
        avg_loss = total_loss / max(step, 1)
        print(f"  Adapter Tuning 完成, 平均 Loss: {avg_loss:.4f}")
        
        if log_file:
            log_file.write(f"  Adapter Tuning: {step} steps, avg Loss = {avg_loss:.4f}\n")
            log_file.flush()
        
        return avg_loss
    
    def train_target(self, source_idx, target_idx, log_file=None):
        """目标域自适应（带 CLIP Adapter）"""
        start_time = datetime.now()
        task_name = f"{self.domains[source_idx]} -> {self.domains[target_idx]}"
        print(f"目标域适应 (带 CLIP Adapter): {task_name}...")
        
        # 构建网络
        self._build_networks()
        
        # 加载 CLIP 模型和 Adapter
        self._load_clip_model()
        self._init_clip_adapter()
        
        # 初始化文本特征
        class_names = self._load_class_names()
        self._init_text_features(class_names)
        
        # 加载源域预训练模型
        source_dir = self._get_output_dir(source_idx)
        self.netG.load_state_dict(torch.load(osp.join(source_dir, "source_G.pt"), map_location=self.device))
        self.netF.load_state_dict(torch.load(osp.join(source_dir, "source_F.pt"), map_location=self.device))
        self.netC.load_state_dict(torch.load(osp.join(source_dir, "source_C.pt"), map_location=self.device))
        self.netD.load_state_dict(torch.load(osp.join(source_dir, "source_D.pt"), map_location=self.device))
        
        # 加载目标域数据
        dset_loaders, num_samples = self._load_target_data(source_idx, target_idx)
        
        # 从配置获取参数
        target_config = self.method_config.get('target', {})
        max_epoch = target_config.get('max_epoch', 15)
        interval = target_config.get('interval', 10)
        
        # 早停配置
        early_stop_patience = target_config.get('early_stop_patience', 3)
        early_stop_enabled = target_config.get('early_stop', True)
        
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
        
        # 写入配置
        self._write_config_to_log(log_file, 'target', source_idx, target_idx)
        
        # 冻结 netC
        self.netC.eval()
        self.netD.train()
        for k, v in self.netC.named_parameters():
            v.requires_grad = False
        
        # 设置 DCA 网络优化器
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
        warmup_iter = self.warmup_epochs * len(dset_loaders["target"])
        
        # 进度条
        pbar = tqdm(total=max_iter, desc=f"Target+Adapter [{task_name}]")
        
        # 早停相关
        best_acc = 0
        no_improve_count = 0
        best_iter = 0
        loss_mix = torch.tensor(0.0).to(self.device)
        
        while iter_num < max_iter:
            try:
                inputs_test, _, tar_idx = next(iter_target)
            except:
                iter_target = iter(dset_loaders["target"])
                inputs_test, _, tar_idx = next(iter_target)
            
            if inputs_test.size(0) == 1:
                continue
            
            # 定期更新伪标签 + Adapter Tuning
            if iter_num % interval_iter == 0 and cls_par > 0:
                self.netG.eval()
                self.netF.eval()
                
                # 更新伪标签
                mem_label1 = obtain_label(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
                mem_label2 = obtain_label_easy(dset_loaders["test"], self.netG, self.netD, device=self.device)
                mem_label1 = torch.from_numpy(mem_label1).to(self.device)
                mem_label2 = torch.from_numpy(mem_label2).to(self.device)
                
                # Adapter Tuning（warmup 后开始）
                if iter_num >= warmup_iter and self.adapter_loss_weight > 0:
                    print(f"\n[Iter {iter_num}] 开始 Adapter Tuning...")
                    high_conf_indices, high_conf_probs = self._collect_high_confidence_samples(dset_loaders["target"])
                    if len(high_conf_indices) > 0:
                        self._tune_adapter(dset_loaders, high_conf_indices, high_conf_probs, log_file)
                    else:
                        print("  警告: 没有高置信度样本，跳过 Adapter Tuning")
                
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
            
            # Step B: 联合训练 netG + netD (带 Adapter 蒸馏)
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
            
            # 不确定性加权
            kl_distance = nn.KLDivLoss(reduction='none')
            log_sm = nn.LogSoftmax(dim=1)
            softmax_out1_stable = torch.clamp(softmax_out1, min=1e-8)
            softmax_out2_stable = torch.clamp(softmax_out2, min=1e-8)
            variance1 = torch.sum(kl_distance(log_sm(outputs1), softmax_out2_stable), dim=1)
            variance2 = torch.sum(kl_distance(log_sm(outputs2), softmax_out1_stable), dim=1)
            
            exp_variance1 = torch.mean(torch.exp(-variance1))
            exp_variance2 = torch.mean(torch.exp(-variance2))
            
            # CLIP(Adapter) → DCA 蒸馏（针对难样本）
            loss_adapter = torch.tensor(0.0).to(self.device)
            
            if iter_num >= warmup_iter and self.adapter_loss_weight > 0:
                # 选择难样本：高熵或高分歧
                entropy1 = -torch.sum(softmax_out1 * torch.log(softmax_out1 + 1e-8), dim=1)
                entropy2 = -torch.sum(softmax_out2 * torch.log(softmax_out2 + 1e-8), dim=1)
                avg_entropy = (entropy1 + entropy2) / 2
                
                max_entropy = np.log(self.class_num)
                entropy_threshold = max_entropy * 0.15  # 调整阈值
                
                discrepancy = torch.sum(SKL(softmax_out1, softmax_out2), dim=1)
                discrepancy_threshold = discrepancy.quantile(0.65)  # top 35%
                
                high_entropy_mask = avg_entropy > entropy_threshold
                high_discrepancy_mask = discrepancy > discrepancy_threshold
                difficult_mask = high_entropy_mask | high_discrepancy_mask
                difficult_indices = torch.where(difficult_mask)[0]
                
                if len(difficult_indices) > 0:
                    # 获取 CLIP(Adapter) 预测
                    with torch.no_grad():
                        clip_probs = self._get_clip_predictions(inputs_test[difficult_indices], use_adapter=True)
                    
                    # 置信度加权
                    clip_confidence = clip_probs.max(dim=1)[0]
                    weight = clip_confidence / clip_confidence.mean()
                    
                    # KL 蒸馏
                    kl_per_sample1 = F.kl_div(log_sm(outputs1[difficult_indices]), clip_probs, reduction='none').sum(dim=1)
                    kl_per_sample2 = F.kl_div(log_sm(outputs2[difficult_indices]), clip_probs, reduction='none').sum(dim=1)
                    
                    loss_adapter = (weight * (kl_per_sample1 + kl_per_sample2)).mean()
            
            total_loss2 += self.adapter_loss_weight * loss_adapter
            
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
            
            # MixUp
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
            
            # 更新参数
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            total_loss2.backward()
            optimizer_g.step()
            optimizer_d.step()
            
            # 定期评估
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                self.netG.eval()
                self.netF.eval()
                
                _, acc_list1, accuracy1 = cal_acc(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
                _, acc_list2, accuracy2 = cal_acc_easy(dset_loaders["test"], self.netG, self.netD, device=self.device)
                
                log_str = f"Iter:{iter_num}/{max_iter}; Acc_c={accuracy1:.2f}%, Acc_d={accuracy2:.2f}%"
                if self.verbose_loss:
                    log_str += f"; Lcls:{loss_cs.item():.4f}; Ladapter:{loss_adapter.item():.4f}; Lmme:{loss_mme.item():.4f}; Lcb:{loss_cb.item():.4f}; Lmix:{loss_mix.item():.4f}"
                
                log_file.write(log_str + "\n")
                log_file.flush()
                
                pbar.set_postfix({'Acc_c': f'{accuracy1:.2f}%', 'Acc_d': f'{accuracy2:.2f}%'})
                
                # 保存最佳模型
                if accuracy1 > best_acc:
                    best_acc = accuracy1
                    best_iter = iter_num
                    no_improve_count = 0
                    if self.config.get('savemodel', True):
                        torch.save(self.netG.state_dict(), osp.join(output_dir, "target_G.pt"))
                        torch.save(self.netF.state_dict(), osp.join(output_dir, "target_F.pt"))
                        torch.save(self.netC.state_dict(), osp.join(output_dir, "target_C.pt"))
                        torch.save(self.netD.state_dict(), osp.join(output_dir, "target_D.pt"))
                        torch.save(self.clip_adapter.state_dict(), osp.join(output_dir, "clip_adapter.pt"))
                        log_file.write(f"  -> 最佳模型已保存 (Acc={accuracy1:.2f}%)\n")
                        log_file.flush()
                else:
                    no_improve_count += 1
                    log_file.write(f"  -> 未提升 ({no_improve_count}/{early_stop_patience})\n")
                    log_file.flush()
                
                # 早停
                if early_stop_enabled and no_improve_count >= early_stop_patience:
                    log_file.write(f"\n早停触发: 连续 {early_stop_patience} 次评估未提升\n")
                    log_file.write(f"最佳准确率: {best_acc:.2f}% @ iter {best_iter}\n")
                    log_file.flush()
                    print(f"\n早停触发! 最佳准确率: {best_acc:.2f}% @ iter {best_iter}")
                    break
                
                self.netG.train()
                self.netF.train()
        
        pbar.close()
        
        # 写入结束时间
        self._write_end_time(log_file, start_time)
        
        print(f"目标域适应完成 (带 CLIP Adapter): {task_name}! 最佳准确率: {best_acc:.2f}%")
        return output_dir
