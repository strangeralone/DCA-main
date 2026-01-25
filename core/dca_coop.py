"""DCA + CoOp 训练器模块

继承 DCA 训练器，使用 CoOp (Context Optimization) 进行提示学习。
与 CLIP 固定提示不同，CoOp 通过学习可优化的提示向量来适应目标域。

核心区别：
- DCA_CLIP: 固定提示 "a photo of a {class}"
- DCA_CoOp: 可学习提示 [V1][V2]...[Vn]{class}
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
from utils.helpers import cal_acc, cal_acc_easy, obtain_label, obtain_label_easy, cal_acc_tta, cal_acc_easy_tta


class TextEncoder(nn.Module):
    """CLIP 文本编码器封装
    
    用于将可学习的提示向量编码为文本特征。
    """
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """前向传播
        
        Args:
            prompts: 经过组合的提示嵌入 [n_cls, n_ctx + 1 + n_suffix, dim]
            tokenized_prompts: tokenized 的提示，用于获取 EOS 位置
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # [n_ctx + 1 + n_suffix, n_cls, dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [n_cls, n_ctx + 1 + n_suffix, dim]
        x = self.ln_final(x).type(self.dtype)

        # 获取 EOS token 位置的特征
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    """可学习的提示向量（对照官方 CoOp 实现）
    
    CoOp 的核心：将固定提示 "a photo of a" 替换为可学习的向量。
    
    结构：[SOS][V1][V2]...[Vn][CLASS][EOS]
    - SOS: 起始 token
    - V1~Vn: 可学习的上下文向量
    - CLASS: 类别名称的嵌入
    - EOS: 结束 token
    
    参考: https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py
    """
    def __init__(self, clip_model, class_names, n_ctx=16, ctx_init="", class_token_position="end"):
        super().__init__()
        n_cls = len(class_names)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 提示向量的维度
        device = next(clip_model.parameters()).device
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.dtype = dtype
        self.class_token_position = class_token_position
        
        # 初始化可学习的上下文向量
        if ctx_init:
            # 使用预定义文本初始化
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1+n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # 随机初始化（与官方一致：使用标准差 0.02）
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)  # 占位符
        
        print(f'初始化上下文: "{prompt_prefix}"')
        print(f"上下文 token 数量: {n_ctx}")
        
        # 设置为可学习参数
        self.ctx = nn.Parameter(ctx_vectors)
        
        # 准备类别名称的嵌入（与官方一致）
        classnames = [name.replace("_", " ") for name in class_names]
        # 官方使用 prompt_prefix + " " + name + "." 格式
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        # Tokenize 所有提示
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # 保存 token 嵌入的各部分
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        # 计算每个类名的 token 长度（用于 middle/front 位置）
        self.name_lens = [len(clip.tokenize(name)[0].nonzero()) - 2 for name in classnames]  # 减去 SOS 和 EOS
    
    def forward(self):
        """生成完整的提示嵌入（对照官方实现）
        
        Returns:
            prompts: [n_cls, context_length, dim]
        """
        ctx = self.ctx  # [n_ctx, dim]
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, n_ctx, dim]
        
        prefix = self.token_prefix  # [n_cls, 1, dim]
        suffix = self.token_suffix  # [n_cls, *, dim]
        
        if self.class_token_position == "end":
            # 标准 CoOp: [SOS][V1]...[Vn][CLASS][EOS]
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        
        elif self.class_token_position == "middle":
            # CLASS 放在中间: [SOS][V1]...[Vn/2][CLASS][Vn/2+1]...[Vn][EOS]
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i_half1 = ctx[i:i+1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i+1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        
        elif self.class_token_position == "front":
            # CLASS 放在前面: [SOS][CLASS][V1]...[Vn][EOS]
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i = ctx[i:i+1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        
        else:
            raise ValueError(f"Unknown class_token_position: {self.class_token_position}")
        
        return prompts


class DCACoOpTrainer(DCATrainer):
    """DCA + CoOp 训练器类
    
    与 DCAClipTrainer 的区别：
    - CLIP 文本特征不再固定，而是通过可学习的提示向量生成
    - 需要额外优化提示向量的参数
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # CoOp 配置
        coop_config = self.method_config.get('coop', {})
        self.clip_model_name = coop_config.get('model', 'ViT-B/16')
        self.n_ctx = coop_config.get('n_ctx', 16)
        self.ctx_init = coop_config.get('ctx_init', '')
        self.class_token_position = coop_config.get('class_token_position', 'end')
        
        # 损失权重
        self.coop_loss_weight = coop_config.get('loss_weight', 0.1)
        self.coop_temperature = coop_config.get('temperature', 100)
        self.coop_entropy_ratio = coop_config.get('entropy_ratio', 0.15)  # 熵阈值 = max_entropy * ratio
        self.coop_top_k_ratio = coop_config.get('top_k_ratio', 0.2)  # 分歧 top k% 为难样本
        self.iid_weight = coop_config.get('iid_weight', 0.1)  # IID Loss 权重
        
        # 模型组件
        self.clip_model = None
        self.prompt_learner = None
        self.text_encoder = None
        
        # 调试配置
        self.debug = config.get('debug', False)
        self.verbose_loss = config.get('verbose_loss', True)
    
    def _load_clip_model(self):
        """加载 CLIP 模型"""
        print(f"加载 CLIP 模型: {self.clip_model_name}...")
        self.clip_model, _ = clip.load(self.clip_model_name, device=self.device, download_root='./vlm')
        
        # 冻结 CLIP 图像编码器
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 创建文本编码器
        self.text_encoder = TextEncoder(self.clip_model)
        
        print("CLIP 模型已加载!")
    
    def _init_prompt_learner(self, class_names):
        """初始化可学习的提示向量"""
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
        """通过可学习的提示生成文本特征（每次调用都会更新）"""
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
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
        
        # 追加 CoOp 配置
        log_file.write("CoOp 配置:\n")
        log_file.write(f"  coop.model: {self.clip_model_name}\n")
        log_file.write(f"  coop.n_ctx: {self.n_ctx}\n")
        log_file.write(f"  coop.ctx_init: {self.ctx_init or '(random)'}\n")
        log_file.write(f"  coop.class_token_position: {self.class_token_position}\n")
        log_file.write(f"  coop.loss_weight: {self.coop_loss_weight}\n")
        log_file.write(f"  coop.entropy_ratio: {self.coop_entropy_ratio}\n")
        log_file.write(f"  coop.top_k_ratio: {self.coop_top_k_ratio}\n")
        log_file.write(f"  debug: {self.debug}\n")
        log_file.write(f"  verbose_loss: {self.verbose_loss}\n")
        log_file.write("=" * 60 + "\n\n")
        log_file.flush()
    
    def _collect_high_confidence_samples(self, loader):
        """收集高置信度样本（用于两阶段 Prompt Tuning）
        
        参考 DIFO: 只使用高置信度样本训练 Prompt，避免噪声干扰。
        
        Returns:
            high_conf_indices: 高置信度样本的索引列表
            high_conf_probs: 对应的 DCA 软标签
            high_conf_images: 对应的图像路径或数据
        """
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        self.netD.eval()
        
        all_probs = []
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
                
                # 平均预测作为软标签
                avg_probs = (softmax_out1 + softmax_out2) / 2
                
                all_probs.append(avg_probs.cpu())
                all_indices.append(idx)
        
        all_probs = torch.cat(all_probs, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        
        # 计算熵，选择低熵（高置信度）样本
        entropy = -torch.sum(all_probs * torch.log(all_probs + 1e-8), dim=1)
        max_entropy = np.log(self.class_num)
        
        # 使用配置的阈值，选择熵低于阈值的样本
        coop_config = self.method_config.get('coop', {})
        high_conf_threshold = coop_config.get('high_conf_threshold', 0.3)
        threshold = max_entropy * high_conf_threshold
        
        high_conf_mask = entropy < threshold
        high_conf_indices = all_indices[high_conf_mask]
        high_conf_probs = all_probs[high_conf_mask]
        
        print(f"  高置信度样本: {len(high_conf_indices)}/{len(all_indices)} "
              f"({100*len(high_conf_indices)/len(all_indices):.1f}%)")
        
        return high_conf_indices, high_conf_probs
    
    def _tune_prompt_only(self, dset_loaders, high_conf_indices, high_conf_probs, log_file=None):
        """独立 Prompt Tuning 阶段（阶段 1）
        
        只训练 Prompt Learner，冻结所有其他网络。
        使用 IID Loss 让 CLIP 输出对齐高置信度样本的 DCA 预测。
        
        Args:
            dset_loaders: 数据加载器字典
            high_conf_indices: 高置信度样本索引
            high_conf_probs: 高置信度样本的 DCA 软标签
            log_file: 日志文件
        """
        coop_config = self.method_config.get('coop', {})
        prompt_tuning_steps = coop_config.get('prompt_tuning_steps', 10)
        prompt_lr = self.config.get('lr', 0.01) * 0.1
        
        print(f"  开始独立 Prompt Tuning ({prompt_tuning_steps} steps)...")
        
        # 创建高置信度样本的索引到软标签的映射
        idx_to_prob = {idx.item(): prob for idx, prob in zip(high_conf_indices, high_conf_probs)}
        
        # 设置 Prompt 优化器
        optimizer_prompt = optim.SGD(
            [{"params": self.prompt_learner.ctx, "lr": prompt_lr}],
            momentum=0.9, weight_decay=5e-4
        )
        
        # 冻结所有网络参数（只训练 Prompt）
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        self.netD.eval()
        
        # 确保 Prompt Learner 可训练
        self.prompt_learner.train()
        for param in self.prompt_learner.parameters():
            param.requires_grad = True
        
        step = 0
        iter_loader = iter(dset_loaders["target"])
        total_iid_loss = 0.0
        
        while step < prompt_tuning_steps:
            try:
                inputs, _, tar_idx = next(iter_loader)
            except StopIteration:
                iter_loader = iter(dset_loaders["target"])
                inputs, _, tar_idx = next(iter_loader)
            
            # 筛选出高置信度样本
            batch_mask = torch.tensor([i.item() in idx_to_prob for i in tar_idx])
            if not batch_mask.any():
                continue
            
            inputs_high_conf = inputs[batch_mask].to(self.device)
            selected_idx = tar_idx[batch_mask]
            
            # 获取对应的 DCA 软标签
            dca_soft_labels = torch.stack([idx_to_prob[i.item()] for i in selected_idx]).to(self.device)
            
            # 获取 CLIP 图像特征（冻结）
            clip_input = F.interpolate(inputs_high_conf, size=(224, 224), mode='bicubic')
            with torch.no_grad():
                image_features = self.clip_model.encode_image(clip_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 获取可学习的文本特征（有梯度）
            text_features = self._get_text_features()
            
            # 计算 CLIP 预测
            clip_logits = image_features @ text_features.T
            clip_probs = F.softmax(clip_logits.float() * self.coop_temperature, dim=1)
            
            # IID Loss: 让 CLIP 输出与 DCA 软标签对齐
            loss_iid = iid_loss(clip_probs, dca_soft_labels)
            
            optimizer_prompt.zero_grad()
            loss_iid.backward()
            optimizer_prompt.step()
            
            total_iid_loss += loss_iid.item()
            step += 1
        
        avg_iid_loss = total_iid_loss / max(step, 1)
        print(f"  Prompt Tuning 完成, 平均 IID Loss: {avg_iid_loss:.4f}")
        
        if log_file:
            log_file.write(f"  Prompt Tuning: {step} steps, avg IID Loss = {avg_iid_loss:.4f}\n")
            log_file.flush()
        
        # 缓存训练后的文本特征
        with torch.no_grad():
            self.cached_text_features = self._get_text_features()
        
        return avg_iid_loss
    
    def train_target(self, source_idx, target_idx, log_file=None):
        """目标域自适应（带 CoOp 提示学习）"""
        start_time = datetime.now()
        task_name = f"{self.domains[source_idx]} -> {self.domains[target_idx]}"
        print(f"目标域适应 (带 CoOp 提示学习): {task_name}...")
        
        # 构建网络
        self._build_networks()
        
        # 加载 CLIP 模型
        self._load_clip_model()
        
        # 初始化可学习的提示向量
        class_names = self._load_class_names()
        self._init_prompt_learner(class_names)
        
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
        
        # 写入配置信息
        self._write_config_to_log(log_file, 'target', source_idx, target_idx)
        
        # 冻结 netC
        self.netC.eval()
        self.netD.train()
        for k, v in self.netC.named_parameters():
            v.requires_grad = False
        
        # 设置优化器（两阶段策略：只包含 DCA 网络，Prompt 在独立阶段训练）
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
        
        # 初始化缓存的文本特征（用于 KL 蒸馏）
        self.cached_text_features = None
        
        iter_num = 0
        iter_target = iter(dset_loaders["target"])
        max_iter = max_epoch * len(dset_loaders["target"])
        interval_iter = max_iter // interval
        
        # 使用 tqdm 进度条
        pbar = tqdm(total=max_iter, desc=f"Target+CoOp [{task_name}]")
        
        # 早停相关变量
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
            
            # 定期更新伪标签 + 两阶段 Prompt Tuning
            if iter_num % interval_iter == 0 and cls_par > 0:
                self.netG.eval()
                self.netF.eval()
                
                # 阶段 1: 更新伪标签
                mem_label1 = obtain_label(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
                mem_label2 = obtain_label_easy(dset_loaders["test"], self.netG, self.netD, device=self.device)
                mem_label1 = torch.from_numpy(mem_label1).to(self.device)
                mem_label2 = torch.from_numpy(mem_label2).to(self.device)
                
                # 阶段 2: 独立 Prompt Tuning（收集高置信度样本 + 训练 Prompt）
                if self.coop_loss_weight > 0:
                    print(f"\n[Iter {iter_num}] 开始两阶段 Prompt Tuning...")
                    high_conf_indices, high_conf_probs = self._collect_high_confidence_samples(dset_loaders["target"])
                    if len(high_conf_indices) > 0:
                        self._tune_prompt_only(dset_loaders, high_conf_indices, high_conf_probs, log_file)
                    else:
                        print("  警告: 没有高置信度样本，跳过 Prompt Tuning")
                        # 如果没有缓存，使用初始文本特征
                        if self.cached_text_features is None:
                            with torch.no_grad():
                                self.cached_text_features = self._get_text_features()
                
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
            
            # Step B: 联合训练 netG + netD + PromptLearner (带 CoOp 蒸馏)
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
            
            # 两阶段 CoOp 蒸馏：使用缓存的文本特征（Prompt 在独立阶段已训练）
            loss_coop = torch.tensor(0.0).to(self.device)
            
            if self.cached_text_features is not None and self.coop_loss_weight > 0:
                # 选择难样本进行 KL 蒸馏
                entropy1 = -torch.sum(softmax_out1 * torch.log(softmax_out1 + 1e-8), dim=1)
                entropy2 = -torch.sum(softmax_out2 * torch.log(softmax_out2 + 1e-8), dim=1)
                avg_entropy = (entropy1 + entropy2) / 2
                
                max_entropy = np.log(self.class_num)
                entropy_threshold = max_entropy * self.coop_entropy_ratio
                
                discrepancy = torch.sum(SKL(softmax_out1, softmax_out2), dim=1)
                discrepancy_threshold = discrepancy.quantile(1 - self.coop_top_k_ratio)
                
                high_entropy_mask = avg_entropy > entropy_threshold
                high_discrepancy_mask = discrepancy > discrepancy_threshold
                difficult_mask = high_entropy_mask | high_discrepancy_mask
                topk_indices = torch.where(difficult_mask)[0]
                
                if len(topk_indices) > 0:
                    clip_input = F.interpolate(inputs_test[topk_indices], size=(224, 224), mode='bicubic')
                    
                    # 获取 CLIP 图像特征（冻结）
                    with torch.no_grad():
                        image_features = self.clip_model.encode_image(clip_input)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        
                        # 使用缓存的文本特征（不计算梯度，Prompt 不更新）
                        clip_logits = image_features @ self.cached_text_features.T
                        clip_probs = F.softmax(clip_logits.float() * self.coop_temperature, dim=1)
                    
                    # KL 蒸馏损失（只训练 DCA 网络，梯度不流向 Prompt）
                    clip_confidence = clip_probs.max(dim=1)[0]
                    weight = clip_confidence / clip_confidence.mean()
                    
                    kl_per_sample1 = F.kl_div(log_sm(outputs1[topk_indices]), clip_probs, reduction='none').sum(dim=1)
                    kl_per_sample2 = F.kl_div(log_sm(outputs2[topk_indices]), clip_probs, reduction='none').sum(dim=1)
                    
                    loss_coop = (weight * (kl_per_sample1 + kl_per_sample2)).mean()
            
            total_loss2 += self.coop_loss_weight * loss_coop
            
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
            
            # 更新 DCA 网络参数（两阶段策略：Prompt 在独立阶段已更新，这里不更新）
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
                    log_str += f"; Lcls:{loss_cs.item():.4f}; Lcoop:{loss_coop.item():.4f}; Lmme:{loss_mme.item():.4f}; Lcb:{loss_cb.item():.4f}; Lmix:{loss_mix.item():.4f}"
                
                log_file.write(log_str + "\n")
                log_file.flush()
                
                pbar.set_postfix({'Acc_c': f'{accuracy1:.2f}%', 'Acc_d': f'{accuracy2:.2f}%'})
                
                # 检查是否提升并保存最佳模型
                if accuracy1 > best_acc:
                    best_acc = accuracy1
                    best_iter = iter_num
                    no_improve_count = 0
                    if self.config.get('savemodel', True):
                        torch.save(self.netG.state_dict(), osp.join(output_dir, "target_G.pt"))
                        torch.save(self.netF.state_dict(), osp.join(output_dir, "target_F.pt"))
                        torch.save(self.netC.state_dict(), osp.join(output_dir, "target_C.pt"))
                        torch.save(self.netD.state_dict(), osp.join(output_dir, "target_D.pt"))
                        torch.save(self.prompt_learner.state_dict(), osp.join(output_dir, "prompt_learner.pt"))
                        log_file.write(f"  -> 最佳模型已保存 (Acc={accuracy1:.2f}%)\n")
                        log_file.flush()
                else:
                    no_improve_count += 1
                    log_file.write(f"  -> 未提升 ({no_improve_count}/{early_stop_patience})\n")
                    log_file.flush()
                
                # 早停检查
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
        
        # TTA 最终评估（加载最佳模型）
        if self.config.get('use_tta', True):
            print("\n进行 TTA 最终评估...")
            log_file.write("\n" + "=" * 60 + "\n")
            log_file.write("🎯 TTA 最终评估\n")
            log_file.write("=" * 60 + "\n")
            
            # 加载最佳模型
            if self.config.get('savemodel', True):
                self.netG.load_state_dict(torch.load(osp.join(output_dir, "target_G.pt"), map_location=self.device))
                self.netF.load_state_dict(torch.load(osp.join(output_dir, "target_F.pt"), map_location=self.device))
                self.netC.load_state_dict(torch.load(osp.join(output_dir, "target_C.pt"), map_location=self.device))
                self.netD.load_state_dict(torch.load(osp.join(output_dir, "target_D.pt"), map_location=self.device))
            
            self.netG.eval()
            self.netF.eval()
            
            # 普通评估
            _, _, acc_normal = cal_acc(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
            
            # TTA 评估
            _, _, acc_tta = cal_acc_tta(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
            
            improvement = acc_tta - acc_normal
            log_file.write(f"普通评估: {acc_normal:.2f}%\n")
            log_file.write(f"TTA 评估: {acc_tta:.2f}%\n")
            log_file.write(f"TTA 提升: {improvement:+.2f}%\n")
            log_file.write("=" * 60 + "\n")
            log_file.flush()
            
            print(f"TTA 评估完成: {acc_normal:.2f}% -> {acc_tta:.2f}% ({improvement:+.2f}%)")
            best_acc = max(best_acc, acc_tta)
        
        print(f"目标域适应完成 (带 CoOp): {task_name}! 最佳准确率: {best_acc:.2f}%")
        return output_dir
