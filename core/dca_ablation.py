"""DCA VLM 消融实验训练器模块

用于进行不同 VLM (Vision-Language Model) 的消融实验。
支持多种 VLM 后端：OpenAI CLIP, OpenCLIP (EVA-CLIP, DFN, MetaCLIP, SigLIP2 等)

核心功能：
- 支持切换不同 VLM 模型进行对比实验
- 统一的训练接口，方便消融研究
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
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    print("警告: open_clip 未安装，仅支持 OpenAI CLIP 后端")

from core.dca import DCATrainer, op_copy, lr_scheduler
from utils.loss import SKL, entropy, adentropy, class_balance, iid_loss
from utils.helpers import cal_acc, cal_acc_easy, obtain_label, obtain_label_easy


# ============== VLM 模型预设配置 ==============
VLM_PRESETS = {
    # OpenAI CLIP (原生)
    'clip-vit-b-32': {'backend': 'clip', 'model': 'ViT-B/32'},
    'clip-vit-b-16': {'backend': 'clip', 'model': 'ViT-B/16'},
    'clip-vit-l-14': {'backend': 'clip', 'model': 'ViT-L/14'},
    'clip-rn50': {'backend': 'clip', 'model': 'RN50'},
    
    # EVA-CLIP (OpenCLIP)
    'eva-clip-vit-b-16': {'backend': 'openclip', 'model': 'EVA02-B-16', 'pretrained': 'merged2b_s8b_b131k'},
    'eva-clip-vit-l-14': {'backend': 'openclip', 'model': 'EVA02-L-14', 'pretrained': 'merged2b_s4b_b131k'},
    
    # DFN (Data Filtering Networks)
    'dfn-vit-b-16': {'backend': 'openclip', 'model': 'ViT-B-16', 'pretrained': 'dfn2b'},
    'dfn-vit-l-14': {'backend': 'openclip', 'model': 'ViT-L-14', 'pretrained': 'dfn2b'},
    'dfn-vit-h-14': {'backend': 'openclip', 'model': 'ViT-H-14', 'pretrained': 'dfn5b'},
    
    # MetaCLIP
    'metaclip-vit-b-16': {'backend': 'openclip', 'model': 'ViT-B-16', 'pretrained': 'metaclip_400m'},
    'metaclip-vit-l-14': {'backend': 'openclip', 'model': 'ViT-L-14', 'pretrained': 'metaclip_400m'},
    'metaclip-vit-h-14': {'backend': 'openclip', 'model': 'ViT-H-14', 'pretrained': 'metaclip_fullcc'},
    
    # SigLIP2 (通过 timm/HuggingFace Hub)
    # 注意: 这些模型需要使用 hf-hub 格式加载
    'siglip2-vit-b-16': {'backend': 'openclip', 'model': 'hf-hub:timm/ViT-B-16-SigLIP2', 'pretrained': ''},
    'siglip2-vit-l-16': {'backend': 'openclip', 'model': 'hf-hub:timm/ViT-L-16-SigLIP2-256', 'pretrained': ''},
    'siglip2-so400m': {'backend': 'openclip', 'model': 'hf-hub:timm/ViT-SO400M-16-SigLIP2-384', 'pretrained': ''},
    
    # LAION 预训练
    'laion-vit-b-32': {'backend': 'openclip', 'model': 'ViT-B-32', 'pretrained': 'laion2b_s34b_b79k'},
    'laion-vit-l-14': {'backend': 'openclip', 'model': 'ViT-L-14', 'pretrained': 'laion2b_s32b_b82k'},
}


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
    """可学习的提示向量（兼容 OpenAI CLIP 和 OpenCLIP）
    
    CoOp 的核心：将固定提示 "a photo of a" 替换为可学习的向量。
    
    结构：[SOS][V1][V2]...[Vn][CLASS][EOS]
    - SOS: 起始 token
    - V1~Vn: 可学习的上下文向量
    - CLASS: 类别名称的嵌入
    - EOS: 结束 token
    
    参考: https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py
    """
    def __init__(self, clip_model, class_names, n_ctx=16, ctx_init="", class_token_position="end",
                 backend='clip', tokenizer=None):
        super().__init__()
        n_cls = len(class_names)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 提示向量的维度
        device = next(clip_model.parameters()).device
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.dtype = dtype
        self.class_token_position = class_token_position
        self.backend = backend
        self.tokenizer = tokenizer  # OpenCLIP 需要外部传入 tokenizer
        
        # 选择 tokenize 函数
        def _tokenize(text):
            if backend == 'openclip' and tokenizer is not None:
                return tokenizer(text)
            else:
                return clip.tokenize(text)
        
        self._tokenize = _tokenize
        
        # 初始化可学习的上下文向量
        if ctx_init:
            # 使用预定义文本初始化
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = _tokenize(ctx_init).to(device)
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
        tokenized_prompts = torch.cat([_tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # 保存 token 嵌入的各部分
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        # 计算每个类名的 token 长度（用于 middle/front 位置）
        self.name_lens = [len(_tokenize(name)[0].nonzero()) - 2 for name in classnames]  # 减去 SOS 和 EOS
    
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


class DCAAblationTrainer(DCATrainer):
    """DCA VLM 消融实验训练器
    
    用于对比不同 VLM 模型在域适应任务上的表现。
    支持 OpenAI CLIP 和 OpenCLIP 后端。
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # CoOp 配置
        coop_config = self.method_config.get('coop', {})
        
        # VLM 后端配置
        self.vlm_preset = coop_config.get('preset', None)  # 使用预设配置
        self.vlm_backend = coop_config.get('backend', 'clip')  # 'clip' 或 'openclip'
        self.vlm_pretrained = coop_config.get('pretrained', None)  # OpenCLIP 预训练权重
        self.clip_model_name = coop_config.get('model', 'ViT-B/16')
        
        # 如果使用预设，覆盖上述配置
        if self.vlm_preset and self.vlm_preset in VLM_PRESETS:
            preset = VLM_PRESETS[self.vlm_preset]
            self.vlm_backend = preset['backend']
            self.clip_model_name = preset['model']
            self.vlm_pretrained = preset.get('pretrained', None)
            print(f"使用 VLM 预设: {self.vlm_preset}")
        
        # CoOp Prompt 配置
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
        self.tokenizer = None  # OpenCLIP tokenizer
        self.preprocess = None  # 图像预处理
        self.image_size = 224  # 默认输入分辨率
        
        # 调试配置
        self.debug = config.get('debug', False)
        self.verbose_loss = config.get('verbose_loss', True)
    
    def _load_clip_model(self):
        """加载 VLM 模型（支持 OpenAI CLIP 和 OpenCLIP）"""
        print(f"加载 VLM 模型: backend={self.vlm_backend}, model={self.clip_model_name}")
        
        if self.vlm_backend == 'openclip':
            if not OPENCLIP_AVAILABLE:
                raise RuntimeError("OpenCLIP 未安装，请运行: pip install open_clip_torch")
            
            # 检查是否是 HuggingFace Hub 格式
            is_hf_hub = self.clip_model_name.startswith('hf-hub:')
            
            if is_hf_hub:
                # HuggingFace Hub 模型（如 SigLIP2）
                print(f"  从 HuggingFace Hub 加载模型...")
                self.clip_model, self.preprocess = open_clip.create_model_from_pretrained(
                    self.clip_model_name,
                    cache_dir='./vlm'
                )
                self.clip_model = self.clip_model.to(self.device)
                self.tokenizer = open_clip.get_tokenizer(self.clip_model_name)
            else:
                # 标准 OpenCLIP 模型
                print(f"  预训练权重: {self.vlm_pretrained}")
                self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                    self.clip_model_name,
                    pretrained=self.vlm_pretrained,
                    device=self.device,
                    cache_dir='./vlm'
                )
                self.tokenizer = open_clip.get_tokenizer(self.clip_model_name)
            
            # 从预处理中获取输入分辨率
            self.image_size = self._get_image_size_from_preprocess()
            print(f"  输入分辨率: {self.image_size}")
        else:
            # 使用原生 OpenAI CLIP
            self.clip_model, self.preprocess = clip.load(
                self.clip_model_name, 
                device=self.device, 
                download_root='./vlm'
            )
            self.tokenizer = None  # 原生 CLIP 使用 clip.tokenize
            self.image_size = 224
        
        # 冻结 VLM 参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 设置为评估模式
        self.clip_model.eval()
        
        # 创建文本编码器
        self.text_encoder = TextEncoder(self.clip_model)
        
        print(f"VLM 模型已加载! (backend={self.vlm_backend})")
    
    def _get_image_size_from_preprocess(self):
        """从预处理 transform 中提取图像分辨率"""
        if self.preprocess is None:
            return 224
        
        # 尝试从 Compose transforms 中获取
        if hasattr(self.preprocess, 'transforms'):
            for t in self.preprocess.transforms:
                if hasattr(t, 'size'):
                    size = t.size
                    return size if isinstance(size, int) else size[0]
        
        # 尝试从 Resize transform 直接获取
        if hasattr(self.preprocess, 'size'):
            size = self.preprocess.size
            return size if isinstance(size, int) else size[0]
        
        return 224  # 默认值
    
    def _init_prompt_learner(self, class_names):
        """初始化可学习的提示向量"""
        print(f"初始化 PromptLearner: n_ctx={self.n_ctx}, position={self.class_token_position}")
        self.prompt_learner = PromptLearner(
            self.clip_model, 
            class_names,
            n_ctx=self.n_ctx,
            ctx_init=self.ctx_init,
            class_token_position=self.class_token_position,
            backend=self.vlm_backend,
            tokenizer=self.tokenizer
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
        
        # 追加 VLM 消融实验配置
        log_file.write("VLM 消融实验配置:\n")
        log_file.write(f"  vlm.preset: {self.vlm_preset or '(无)'}\n")
        log_file.write(f"  vlm.backend: {self.vlm_backend}\n")
        log_file.write(f"  vlm.model: {self.clip_model_name}\n")
        log_file.write(f"  vlm.pretrained: {self.vlm_pretrained or '(无)'}\n")
        log_file.write(f"  vlm.image_size: {self.image_size}\n")
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
        
        # 设置优化器（包含 DCA 网络 + 提示向量）
        param_group_g = []
        param_group_d = []
        param_group_prompt = []
        
        for k, v in self.netG.named_parameters():
            param_group_g += [{"params": v, "lr": lr * 0.1}]
        for k, v in self.netF.named_parameters():
            param_group_g += [{"params": v, "lr": lr * 1.0}]
        for k, v in self.netD.named_parameters():
            param_group_d += [{"params": v, "lr": lr * 2.0}]
        
        # 提示向量优化器（使用较小的学习率）
        param_group_prompt = [{"params": self.prompt_learner.ctx, "lr": lr * 0.5}]
        
        optimizer_g = optim.SGD(param_group_g)
        optimizer_d = optim.SGD(param_group_d)
        optimizer_prompt = optim.SGD(param_group_prompt)
        
        optimizer_g = op_copy(optimizer_g)
        optimizer_d = op_copy(optimizer_d)
        optimizer_prompt = op_copy(optimizer_prompt)
        
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
            lr_scheduler(optimizer_prompt, iter_num=iter_num, max_iter=max_iter)
            
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
            
            # CoOp 蒸馏损失 - 难样本选择
            # 计算两个分类器的熵（不确定性）
            entropy1 = -torch.sum(softmax_out1 * torch.log(softmax_out1 + 1e-8), dim=1)
            entropy2 = -torch.sum(softmax_out2 * torch.log(softmax_out2 + 1e-8), dim=1)
            avg_entropy = (entropy1 + entropy2) / 2
            
            # 基于最大熵的阈值：max_entropy = ln(class_num)
            max_entropy = np.log(self.class_num)
            entropy_threshold = max_entropy * self.coop_entropy_ratio
            
            # 基于分歧的阈值（使用百分位数）
            discrepancy = torch.sum(SKL(softmax_out1, softmax_out2), dim=1)
            discrepancy_threshold = discrepancy.quantile(1 - self.coop_top_k_ratio)
            
            # 难样本 = 高熵 OR 高分歧
            high_entropy_mask = avg_entropy > entropy_threshold
            high_discrepancy_mask = discrepancy > discrepancy_threshold
            difficult_mask = high_entropy_mask | high_discrepancy_mask
            topk_indices = torch.where(difficult_mask)[0]
            
            if len(topk_indices) > 0:
                clip_input = F.interpolate(inputs_test[topk_indices], size=(self.image_size, self.image_size), mode='bicubic')
                
                # 获取 CLIP 图像特征
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(clip_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 获取可学习的文本特征（这里会有梯度，用于 IID Loss）
                text_features = self._get_text_features()
                
                # 计算 CLIP 相似度
                clip_logits = image_features @ text_features.T
                clip_probs = F.softmax(clip_logits.float() * self.coop_temperature, dim=1)
                
                # DCA 软标签作为 prompt 训练目标
                dca_soft_labels = ((softmax_out1 + softmax_out2) / 2)[topk_indices].detach()
                
                # 1. KL 蒸馏损失：基于 CLIP 置信度加权（归一化权重，保持 loss 量级不变）
                clip_confidence = clip_probs.max(dim=1)[0]  # [num_difficult_samples]
                weight = clip_confidence / clip_confidence.mean()  # 归一化：高置信度放大，低置信度缩小
                
                # 按样本计算 KL（reduction='none' 保留每个样本的损失）
                kl_per_sample1 = F.kl_div(log_sm(outputs1[topk_indices]), clip_probs.detach(), reduction='none').sum(dim=1)
                kl_per_sample2 = F.kl_div(log_sm(outputs2[topk_indices]), clip_probs.detach(), reduction='none').sum(dim=1)
                
                # 归一化权重加权求平均
                loss_kl = (weight * (kl_per_sample1 + kl_per_sample2)).mean()
                
                # 2. IID 损失：训练 Prompt 最大化 CLIP 与 DCA 的互信息（梯度流向 prompt）
                loss_iid = iid_loss(clip_probs, dca_soft_labels)
                
                loss_coop = loss_kl + self.iid_weight * loss_iid
            else:
                loss_coop = torch.tensor(0.0).to(self.device)
                loss_iid = torch.tensor(0.0).to(self.device)
            
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
            
            # 更新所有参数
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            optimizer_prompt.zero_grad()
            total_loss2.backward()
            optimizer_g.step()
            optimizer_d.step()
            optimizer_prompt.step()
            
            # 定期评估
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                self.netG.eval()
                self.netF.eval()
                
                _, acc_list1, accuracy1 = cal_acc(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
                _, acc_list2, accuracy2 = cal_acc_easy(dset_loaders["test"], self.netG, self.netD, device=self.device)
                
                log_str = f"Iter:{iter_num}/{max_iter}; Acc_c={accuracy1:.2f}%, Acc_d={accuracy2:.2f}%"
                if self.verbose_loss:
                    log_str += f"; Lcls:{loss_cs.item():.4f}; Lcoop:{loss_coop.item():.4f}; Liid:{loss_iid.item():.4f}; Lmme:{loss_mme.item():.4f}; Lcb:{loss_cb.item():.4f}; Lmix:{loss_mix.item():.4f}"
                
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
        
        print(f"目标域适应完成 (带 CoOp): {task_name}! 最佳准确率: {best_acc:.2f}%")
        return output_dir
