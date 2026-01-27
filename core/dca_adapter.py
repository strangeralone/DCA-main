"""DCA + CoOp + CLIP Adapter 训练器模块

继承 DCA 训练器，同时使用：
- CoOp (Context Optimization): 文本端可学习 Prompt
- CLIP Adapter: 视觉端适应

核心创新：双端适应 + 双向知识迁移
- 文本端：CoOp 学习适应目标域的 prompt
- 视觉端：Adapter 适应目标域的视觉特征
- 双向：DCA ↔ CLIP(CoOp + Adapter)

参考：
- CoOp: Learning to Prompt for Vision-Language Models (IJCV 2022)
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

# 从 dca_coop 导入 Prompt Learning 相关类
from core.dca_coop import TextEncoder, PromptLearner


class DCAAdapterTrainer(DCATrainer):
    """DCA + CoOp + CLIP Adapter 训练器类
    
    双端适应：
    1. 文本端：CoOp 可学习的 prompt
    2. 视觉端：轻量 Adapter
    3. 双向蒸馏：DCA ↔ CLIP(CoOp + Adapter)
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
        self.adapter_loss_weight = adapter_config.get('loss_weight', 0.5)  # 提高权重
        
        # CoOp 配置
        self.n_ctx = adapter_config.get('n_ctx', 4)
        self.ctx_init = adapter_config.get('ctx_init', 'a_photo_of_a')
        self.class_token_position = adapter_config.get('class_token_position', 'end')
        self.prompt_lr = adapter_config.get('prompt_lr', 0.001)
        
        # 高置信度样本选择配置
        self.high_conf_threshold = adapter_config.get('high_conf_threshold', 0.3)
        self.consistency_threshold = adapter_config.get('consistency_threshold', 0.8)
        self.entropy_ratio = adapter_config.get('entropy_ratio', 0.15)
        self.top_k_ratio = adapter_config.get('top_k_ratio', 0.35)
        
        # 训练配置（去掉 warmup）
        self.tuning_steps = adapter_config.get('tuning_steps', 20)
        
        # 模型组件
        self.clip_model = None
        self.clip_adapter = None
        self.prompt_learner = None
        self.text_encoder = None
        self.cached_text_features = None
        
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
        
        # 创建文本编码器
        self.text_encoder = TextEncoder(self.clip_model)
        
        print("CLIP 模型已加载!")
    
    def _init_clip_adapter(self):
        """初始化 CLIP 视觉 Adapter"""
        # 获取 CLIP 输出维度
        if 'ViT-B' in self.clip_model_name:
            clip_dim = 512
        elif 'ViT-L' in self.clip_model_name:
            clip_dim = 768
        else:
            clip_dim = 512
        
        print(f"初始化 CLIP Adapter: dim={clip_dim}, bottleneck={self.adapter_bottleneck}, "
              f"residual={self.adapter_residual_ratio}")
        
        self.clip_adapter = CLIPAdapter(
            in_dim=clip_dim,
            bottleneck_ratio=self.adapter_bottleneck,
            residual_ratio=self.adapter_residual_ratio
        ).to(self.device)
        
        num_params = sum(p.numel() for p in self.clip_adapter.parameters() if p.requires_grad)
        print(f"CLIP Adapter 可训练参数: {num_params:,}")
    
    def _init_prompt_learner(self, class_names):
        """初始化可学习的提示向量（CoOp）"""
        print(f"初始化 PromptLearner: n_ctx={self.n_ctx}, position={self.class_token_position}")
        self.prompt_learner = PromptLearner(
            self.clip_model, 
            class_names,
            n_ctx=self.n_ctx,
            ctx_init=self.ctx_init,
            class_token_position=self.class_token_position
        ).to(self.device)
        print(f"可学习的提示向量形状: {self.prompt_learner.ctx.shape}")
    
    def _get_text_features(self):
        """通过可学习的提示生成文本特征"""
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def _get_adapted_clip_features(self, images):
        """获取经过 Adapter 的 CLIP 图像特征"""
        # 调整图像大小到 CLIP 输入尺寸
        clip_input = F.interpolate(images, size=(224, 224), mode='bicubic')
        
        # 获取 CLIP 图像特征（冻结）
        with torch.no_grad():
            image_features = self.clip_model.encode_image(clip_input)
        
        # 应用 Adapter（可学习）
        image_features = self.clip_adapter(image_features.float())
        
        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    def _get_clip_predictions(self, images, text_features=None):
        """获取 CLIP 预测概率（使用 CoOp + Adapter）"""
        image_features = self._get_adapted_clip_features(images)
        
        if text_features is None:
            text_features = self._get_text_features()
        
        logits = image_features @ text_features.T
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
        
        # 追加配置
        log_file.write("CoOp + Adapter 配置:\n")
        log_file.write(f"  model: {self.clip_model_name}\n")
        log_file.write(f"  adapter.bottleneck_ratio: {self.adapter_bottleneck}\n")
        log_file.write(f"  adapter.residual_ratio: {self.adapter_residual_ratio}\n")
        log_file.write(f"  adapter.adapter_lr: {self.adapter_lr}\n")
        log_file.write(f"  coop.n_ctx: {self.n_ctx}\n")
        log_file.write(f"  coop.ctx_init: {self.ctx_init}\n")
        log_file.write(f"  coop.prompt_lr: {self.prompt_lr}\n")
        log_file.write(f"  temperature: {self.adapter_temperature}\n")
        log_file.write(f"  loss_weight: {self.adapter_loss_weight}\n")
        log_file.write(f"  high_conf_threshold: {self.high_conf_threshold}\n")
        log_file.write(f"  tuning_steps: {self.tuning_steps}\n")
        log_file.write("=" * 60 + "\n\n")
        log_file.flush()
    
    def _collect_high_confidence_samples(self, loader):
        """收集高置信度样本（用于 CoOp + Adapter 训练）"""
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
        
        avg_probs = (all_probs_c + all_probs_d) / 2
        
        # 低熵
        entropy_vals = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8), dim=1)
        max_entropy = np.log(self.class_num)
        entropy_threshold = max_entropy * self.high_conf_threshold
        low_entropy_mask = entropy_vals < entropy_threshold
        
        # 双分类器一致
        pred_c = all_probs_c.argmax(dim=1)
        pred_d = all_probs_d.argmax(dim=1)
        consistency_mask = pred_c == pred_d
        
        # 综合
        high_conf_mask = low_entropy_mask & consistency_mask
        
        high_conf_indices = all_indices[high_conf_mask]
        high_conf_probs = avg_probs[high_conf_mask]
        
        print(f"  高置信度样本: {len(high_conf_indices)}/{len(all_indices)} "
              f"({100*len(high_conf_indices)/len(all_indices):.1f}%)")
        
        return high_conf_indices, high_conf_probs
    
    def _tune_coop_and_adapter(self, dset_loaders, high_conf_indices, high_conf_probs, log_file=None):
        """独立 CoOp + Adapter Tuning 阶段
        
        同时训练 Prompt 和 Adapter，用 DCA 高置信度预测指导。
        """
        print(f"  开始 CoOp + Adapter Tuning ({self.tuning_steps} steps)...")
        
        idx_to_prob = {idx.item(): prob for idx, prob in zip(high_conf_indices, high_conf_probs)}
        
        # 联合优化器：Prompt + Adapter
        optimizer = optim.AdamW([
            {"params": self.prompt_learner.ctx, "lr": self.prompt_lr},
            {"params": self.clip_adapter.parameters(), "lr": self.adapter_lr}
        ], weight_decay=0.01)
        
        # 冻结 DCA
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        self.netD.eval()
        
        # 训练 Prompt + Adapter
        self.prompt_learner.train()
        self.clip_adapter.train()
        
        step = 0
        iter_loader = iter(dset_loaders["target"])
        total_loss = 0.0
        
        while step < self.tuning_steps:
            try:
                inputs, _, tar_idx = next(iter_loader)
            except StopIteration:
                iter_loader = iter(dset_loaders["target"])
                inputs, _, tar_idx = next(iter_loader)
            
            batch_mask = torch.tensor([i.item() in idx_to_prob for i in tar_idx])
            if not batch_mask.any():
                continue
            
            inputs_high_conf = inputs[batch_mask].to(self.device)
            selected_idx = tar_idx[batch_mask]
            
            dca_soft_labels = torch.stack([idx_to_prob[i.item()] for i in selected_idx]).to(self.device)
            
            # 获取 CLIP(CoOp + Adapter) 预测
            clip_probs = self._get_clip_predictions(inputs_high_conf)
            
            # IID Loss
            loss = iid_loss(clip_probs, dca_soft_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
        
        avg_loss = total_loss / max(step, 1)
        print(f"  CoOp + Adapter Tuning 完成, 平均 Loss: {avg_loss:.4f}")
        
        if log_file:
            log_file.write(f"  CoOp + Adapter Tuning: {step} steps, avg Loss = {avg_loss:.4f}\n")
            log_file.flush()
        
        # 缓存训练后的文本特征
        with torch.no_grad():
            self.cached_text_features = self._get_text_features()
        
        return avg_loss
    
    def train_target(self, source_idx, target_idx, log_file=None):
        """目标域自适应（带 CoOp + Adapter）"""
        start_time = datetime.now()
        task_name = f"{self.domains[source_idx]} -> {self.domains[target_idx]}"
        print(f"目标域适应 (CoOp + Adapter 双端适应): {task_name}...")
        
        # 构建网络
        self._build_networks()
        
        # 加载 CLIP
        self._load_clip_model()
        self._init_clip_adapter()
        
        # 初始化 Prompt Learner
        class_names = self._load_class_names()
        self._init_prompt_learner(class_names)
        
        # 初始缓存文本特征
        with torch.no_grad():
            self.cached_text_features = self._get_text_features()
        
        # 加载源域预训练模型
        source_dir = self._get_output_dir(source_idx)
        self.netG.load_state_dict(torch.load(osp.join(source_dir, "source_G.pt"), map_location=self.device))
        self.netF.load_state_dict(torch.load(osp.join(source_dir, "source_F.pt"), map_location=self.device))
        self.netC.load_state_dict(torch.load(osp.join(source_dir, "source_C.pt"), map_location=self.device))
        self.netD.load_state_dict(torch.load(osp.join(source_dir, "source_D.pt"), map_location=self.device))
        
        # 加载数据
        dset_loaders, num_samples = self._load_target_data(source_idx, target_idx)
        
        # 配置
        target_config = self.method_config.get('target', {})
        max_epoch = target_config.get('max_epoch', 15)
        interval = target_config.get('interval', 10)
        early_stop_patience = target_config.get('early_stop_patience', 3)
        early_stop_enabled = target_config.get('early_stop', True)
        
        lamda = self.method_config.get('lamda', 0.45)
        cls_par = self.method_config.get('cls_par', 0.15)
        alpha = self.method_config.get('alpha', 0.5)
        mix = self.method_config.get('mix', 0.5)
        lr = self.config.get('lr', 0.01)
        
        output_dir = self._get_output_dir(source_idx, target_idx)
        os.makedirs(output_dir, exist_ok=True)
        
        if log_file is None:
            log_file = open(osp.join(output_dir, "log.txt"), "w")
        
        self._write_config_to_log(log_file, 'target', source_idx, target_idx)
        
        # 冻结 netC
        self.netC.eval()
        self.netD.train()
        for k, v in self.netC.named_parameters():
            v.requires_grad = False
        
        # DCA 优化器
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
        
        pbar = tqdm(total=max_iter, desc=f"Target+CoOp+Adapter [{task_name}]")
        
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
            
            # 定期更新（从第一轮开始，去掉 warmup）
            if iter_num % interval_iter == 0 and cls_par > 0:
                self.netG.eval()
                self.netF.eval()
                
                # 更新伪标签
                mem_label1 = obtain_label(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
                mem_label2 = obtain_label_easy(dset_loaders["test"], self.netG, self.netD, device=self.device)
                mem_label1 = torch.from_numpy(mem_label1).to(self.device)
                mem_label2 = torch.from_numpy(mem_label2).to(self.device)
                
                # CoOp + Adapter Tuning（从第一轮开始）
                if self.adapter_loss_weight > 0:
                    print(f"\n[Iter {iter_num}] 开始 CoOp + Adapter Tuning...")
                    high_conf_indices, high_conf_probs = self._collect_high_confidence_samples(dset_loaders["target"])
                    if len(high_conf_indices) > 0:
                        self._tune_coop_and_adapter(dset_loaders, high_conf_indices, high_conf_probs, log_file)
                    else:
                        print("  警告: 没有高置信度样本，跳过 Tuning")
                
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
            
            # Step B: 联合训练 (带 CLIP 蒸馏)
            total_loss2 = 0
            features_d = self.netG(inputs_test)
            features = self.netF(features_d)
            outputs1 = self.netC(features)
            outputs2 = self.netD(features_d)
            
            softmax_out1 = nn.Softmax(dim=1)(outputs1)
            softmax_out2 = nn.Softmax(dim=1)(outputs2)
            
            pred1 = mem_label1[tar_idx]
            pred2 = mem_label2[tar_idx]
            
            classifier_loss1 = nn.CrossEntropyLoss()(outputs1, pred1)
            classifier_loss2 = nn.CrossEntropyLoss()(outputs2, pred2)
            
            kl_distance = nn.KLDivLoss(reduction='none')
            log_sm = nn.LogSoftmax(dim=1)
            softmax_out1_stable = torch.clamp(softmax_out1, min=1e-8)
            softmax_out2_stable = torch.clamp(softmax_out2, min=1e-8)
            variance1 = torch.sum(kl_distance(log_sm(outputs1), softmax_out2_stable), dim=1)
            variance2 = torch.sum(kl_distance(log_sm(outputs2), softmax_out1_stable), dim=1)
            
            exp_variance1 = torch.mean(torch.exp(-variance1))
            exp_variance2 = torch.mean(torch.exp(-variance2))
            
            # CLIP(CoOp + Adapter) → DCA 蒸馏（针对难样本，从第一轮开始）
            loss_clip = torch.tensor(0.0).to(self.device)
            
            if self.cached_text_features is not None and self.adapter_loss_weight > 0:
                # 选择难样本
                entropy1 = -torch.sum(softmax_out1 * torch.log(softmax_out1 + 1e-8), dim=1)
                entropy2 = -torch.sum(softmax_out2 * torch.log(softmax_out2 + 1e-8), dim=1)
                avg_entropy = (entropy1 + entropy2) / 2
                
                max_entropy = np.log(self.class_num)
                entropy_threshold = max_entropy * self.entropy_ratio
                
                discrepancy = torch.sum(SKL(softmax_out1, softmax_out2), dim=1)
                discrepancy_threshold = discrepancy.quantile(1 - self.top_k_ratio)
                
                high_entropy_mask = avg_entropy > entropy_threshold
                high_discrepancy_mask = discrepancy > discrepancy_threshold
                difficult_mask = high_entropy_mask | high_discrepancy_mask
                difficult_indices = torch.where(difficult_mask)[0]
                
                if len(difficult_indices) > 0:
                    # 冻结 Prompt + Adapter，只蒸馏
                    with torch.no_grad():
                        image_features = self._get_adapted_clip_features(inputs_test[difficult_indices])
                        clip_logits = image_features @ self.cached_text_features.T
                        clip_probs = F.softmax(clip_logits.float() * self.adapter_temperature, dim=1)
                    
                    clip_confidence = clip_probs.max(dim=1)[0]
                    weight = clip_confidence / clip_confidence.mean()
                    
                    kl_per_sample1 = F.kl_div(log_sm(outputs1[difficult_indices]), clip_probs, reduction='none').sum(dim=1)
                    kl_per_sample2 = F.kl_div(log_sm(outputs2[difficult_indices]), clip_probs, reduction='none').sum(dim=1)
                    
                    loss_clip = (weight * (kl_per_sample1 + kl_per_sample2)).mean()
            
            total_loss2 += self.adapter_loss_weight * loss_clip
            
            # 分类损失
            loss_seg1 = classifier_loss1 * exp_variance1 + torch.mean(variance1)
            loss_seg2 = classifier_loss2 * exp_variance2 + torch.mean(variance2)
            classifier_loss = alpha * loss_seg1 + (2 - alpha) * loss_seg2
            loss_cs = cls_par * classifier_loss
            total_loss2 += loss_cs
            
            # 对抗熵
            loss_ent1 = adentropy(self.netC, features, lamda)
            loss_ent2 = adentropy(self.netD, features_d, lamda)
            loss_mme = loss_ent1 + loss_ent2
            total_loss2 += loss_mme
            
            # 类平衡
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
            
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            total_loss2.backward()
            optimizer_g.step()
            optimizer_d.step()
            
            # 评估
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                self.netG.eval()
                self.netF.eval()
                
                _, acc_list1, accuracy1 = cal_acc(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
                _, acc_list2, accuracy2 = cal_acc_easy(dset_loaders["test"], self.netG, self.netD, device=self.device)
                
                log_str = f"Iter:{iter_num}/{max_iter}; Acc_c={accuracy1:.2f}%, Acc_d={accuracy2:.2f}%"
                if self.verbose_loss:
                    log_str += f"; Lcls:{loss_cs.item():.4f}; Lclip:{loss_clip.item():.4f}; Lmme:{loss_mme.item():.4f}; Lcb:{loss_cb.item():.4f}; Lmix:{loss_mix.item():.4f}"
                
                log_file.write(log_str + "\n")
                log_file.flush()
                
                pbar.set_postfix({'Acc_c': f'{accuracy1:.2f}%', 'Acc_d': f'{accuracy2:.2f}%'})
                
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
                        torch.save(self.prompt_learner.state_dict(), osp.join(output_dir, "prompt_learner.pt"))
                        log_file.write(f"  -> 最佳模型已保存 (Acc={accuracy1:.2f}%)\n")
                        log_file.flush()
                else:
                    no_improve_count += 1
                    log_file.write(f"  -> 未提升 ({no_improve_count}/{early_stop_patience})\n")
                    log_file.flush()
                
                if early_stop_enabled and no_improve_count >= early_stop_patience:
                    log_file.write(f"\n早停触发: 连续 {early_stop_patience} 次评估未提升\n")
                    log_file.write(f"最佳准确率: {best_acc:.2f}% @ iter {best_iter}\n")
                    log_file.flush()
                    print(f"\n早停触发! 最佳准确率: {best_acc:.2f}% @ iter {best_iter}")
                    break
                
                self.netG.train()
                self.netF.train()
        
        pbar.close()
        self._write_end_time(log_file, start_time)
        
        print(f"目标域适应完成 (CoOp + Adapter): {task_name}! 最佳准确率: {best_acc:.2f}%")
        return output_dir
