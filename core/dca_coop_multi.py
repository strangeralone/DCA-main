"""DCA + CoOp Multi-Template 训练器模块

继承 DCACoOpTrainer，使用多个可学习的提示模板，每个模板有独立的上下文向量。
最终文本特征通过融合多个模板的特征得到。
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
from core.dca_coop import TextEncoder
from utils.loss import SKL, entropy, adentropy, class_balance, iid_loss
from utils.helpers import cal_acc, cal_acc_easy, obtain_label, obtain_label_easy


class MultiPromptLearner(nn.Module):
    """多组可学习提示向量
    
    与标准 CoOp 相同的结构 [SOS][V1][V2]...[Vn][CLASS][.][EOS]，
    但有多组独立的可学习上下文向量，每组用不同模板初始化。
    最终融合所有组的文本特征。
    """
    def __init__(self, clip_model, class_names, ctx_inits, class_token_position="end"):
        """
        Args:
            clip_model: CLIP 模型
            class_names: 类别名称列表
            ctx_inits: 初始化模板列表，如 ["a photo of a", "an image of a", ...]
            class_token_position: 类别 token 位置
        """
        super().__init__()
        n_cls = len(class_names)
        n_prompts = len(ctx_inits)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        device = next(clip_model.parameters()).device
        
        self.n_cls = n_cls
        self.n_prompts = n_prompts
        self.dtype = dtype
        self.class_token_position = class_token_position
        
        print(f"初始化 {n_prompts} 组可学习提示向量...")
        
        # 解析每个模板的 n_ctx（token 数量可能不同）
        n_ctx_list = []
        ctx_vectors_list = []
        
        for i, ctx_init in enumerate(ctx_inits):
            ctx_init = ctx_init.replace("_", " ")
            tokens = ctx_init.split(" ")
            n_ctx = len(tokens)
            n_ctx_list.append(n_ctx)
            
            # 用模板文本初始化可学习向量
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1+n_ctx, :]  # 跳过 SOS
            ctx_vectors_list.append(ctx_vectors)
            
            print(f"  组 {i}: \"{ctx_init}\" -> {n_ctx} tokens")
        
        # 检查所有组的 n_ctx 是否相同
        if len(set(n_ctx_list)) > 1:
            raise ValueError(f"所有模板的 token 数量必须相同，当前: {n_ctx_list}")
        
        self.n_ctx = n_ctx_list[0]
        
        # 合并为可学习参数 [n_prompts, n_ctx, ctx_dim]
        ctx_vectors = torch.stack(ctx_vectors_list, dim=0)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # 准备类别名称（标准 CoOp 格式）
        classnames = [name.replace("_", " ") for name in class_names]
        prompt_prefix = " ".join(["X"] * self.n_ctx)
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        # Tokenize
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # 标准 CoOp 结构: [SOS] + [可学习 V1...Vn] + [CLASS][.][EOS]
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLASS + . + EOS
        self.tokenized_prompts = tokenized_prompts
        
        print(f"多组可学习提示初始化完成: {n_prompts} 组, 每组 {self.n_ctx} 个 token")
    
    def forward(self, prompt_idx=None):
        """生成提示嵌入"""
        if prompt_idx is not None:
            return self._forward_single(prompt_idx)
        else:
            return [self._forward_single(i) for i in range(self.n_prompts)]
    
    def _forward_single(self, p_idx):
        """生成单组提示嵌入"""
        ctx = self.ctx[p_idx]  # [n_ctx, dim]
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, n_ctx, dim]
        
        # 标准结构: [SOS][V1...Vn][CLASS][.][EOS]
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts


class DCACoOpMultiTrainer(DCATrainer):
    """DCA + CoOp Multi-Prompt 训练器类
    
    使用多组独立的可学习提示向量，每组用不同模板初始化，最终融合。
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        coop_config = self.method_config.get('coop_multi', {})
        self.clip_model_name = coop_config.get('model', 'ViT-B/16')
        self.coop_loss_weight = coop_config.get('loss_weight', 1.0)
        self.coop_temperature = coop_config.get('temperature', 100)
        self.coop_entropy_ratio = coop_config.get('entropy_ratio', 0.15)
        self.coop_top_k_ratio = coop_config.get('top_k_ratio', 0.2)
        self.iid_weight = coop_config.get('iid_weight', 0.1)  # IID Loss 权重
        
        # 多组模板初始化配置
        self.ctx_inits = coop_config.get('ctx_inits', [
            "a photo of a",
            "an image of a",
            "a picture of a",
            "a good photo of a",
        ])
        self.class_token_position = coop_config.get('class_token_position', 'end')
        
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
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.text_encoder = TextEncoder(self.clip_model)
        print("CLIP 模型已加载!")
    
    def _init_prompt_learner(self, class_names):
        """初始化多组可学习提示向量"""
        print(f"初始化 MultiPromptLearner: {len(self.ctx_inits)} 组")
        self.prompt_learner = MultiPromptLearner(
            self.clip_model,
            class_names,
            ctx_inits=self.ctx_inits,
            class_token_position=self.class_token_position
        ).to(self.device)
    
    def _get_text_features(self):
        """获取融合后的文本特征（多组平均）"""
        all_features = []
        
        for p_idx in range(self.prompt_learner.n_prompts):
            prompts = self.prompt_learner(p_idx)
            features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features)
        
        # 平均融合
        stacked = torch.stack(all_features, dim=0)
        text_features = stacked.mean(dim=0)
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
        
        log_file.write("CoOp Multi-Prompt 配置:\n")
        log_file.write(f"  model: {self.clip_model_name}\n")
        log_file.write(f"  ctx_inits: {self.ctx_inits}\n")
        log_file.write(f"  loss_weight: {self.coop_loss_weight}\n")
        log_file.write(f"  iid_weight: {self.iid_weight}\n")
        log_file.write(f"  entropy_ratio: {self.coop_entropy_ratio}\n")
        log_file.write(f"  top_k_ratio: {self.coop_top_k_ratio}\n")
        log_file.write(f"  temperature: {self.coop_temperature}\n")
        log_file.write("=" * 60 + "\n\n")
        log_file.flush()
    
    def train_target(self, source_idx, target_idx, log_file=None):
        """目标域自适应（带多模板 CoOp）"""
        start_time = datetime.now()
        task_name = f"{self.domains[source_idx]} -> {self.domains[target_idx]}"
        print(f"目标域适应 (CoOp Multi-Template): {task_name}...")
        
        # 构建网络
        self._build_networks()
        
        # 加载 CLIP 模型
        self._load_clip_model()
        
        # 初始化多模板可学习提示向量
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
        
        # 多模板提示优化器
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
        
        pbar = tqdm(total=max_iter, desc=f"Target+CoOpMulti [{task_name}]", ncols=120)
        
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
            
            # 更新伪标签
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
            
            # Step B: 联合训练
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
            
            # 难样本选择
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
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(clip_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 获取多模板融合的文本特征（有梯度，用于 IID Loss）
                text_features = self._get_text_features()
                
                clip_logits = image_features @ text_features.T
                clip_probs = F.softmax(clip_logits.float() * self.coop_temperature, dim=1)
                
                # DCA 软标签作为 prompt 训练目标
                dca_soft_labels = ((softmax_out1 + softmax_out2) / 2)[topk_indices].detach()
                
                # 1. KL 蒸馏损失：训练 DCA 网络向 CLIP 对齐（detach CLIP 输出）
                loss_kl = F.kl_div(log_sm(outputs1[topk_indices]), clip_probs.detach(), reduction='batchmean') + \
                          F.kl_div(log_sm(outputs2[topk_indices]), clip_probs.detach(), reduction='batchmean')
                
                # 2. IID 损失：训练 Prompt 最大化 CLIP 与 DCA 的互信息（梯度流向 prompt）
                loss_iid = iid_loss(clip_probs, dca_soft_labels)
                
                loss_coop = loss_kl + self.iid_weight * loss_iid
            else:
                loss_coop = torch.tensor(0.0).to(self.device)
                loss_iid = torch.tensor(0.0).to(self.device)
            
            total_loss2 += self.coop_loss_weight * loss_coop
            
            loss_seg1 = classifier_loss1 * exp_variance1 + torch.mean(variance1)
            loss_seg2 = classifier_loss2 * exp_variance2 + torch.mean(variance2)
            classifier_loss = alpha * loss_seg1 + (2 - alpha) * loss_seg2
            loss_cs = cls_par * classifier_loss
            total_loss2 += loss_cs
            
            loss_ent1 = adentropy(self.netC, features, lamda)
            loss_ent2 = adentropy(self.netD, features_d, lamda)
            loss_mme = loss_ent1 + loss_ent2
            total_loss2 += loss_mme
            
            loss_cb1 = class_balance(softmax_out1, lamda)
            loss_cb2 = class_balance(softmax_out2, lamda)
            loss_cb = loss_cb1 + loss_cb2
            total_loss2 += loss_cb
            
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
            optimizer_prompt.zero_grad()
            total_loss2.backward()
            optimizer_g.step()
            optimizer_d.step()
            optimizer_prompt.step()
            
            # 评估
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                self.netG.eval()
                self.netF.eval()
                
                _, acc_list1, accuracy1 = cal_acc(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
                _, acc_list2, accuracy2 = cal_acc_easy(dset_loaders["test"], self.netG, self.netD, device=self.device)
                
                log_str = f"Iter:{iter_num}/{max_iter}; Acc_c={accuracy1:.2f}%, Acc_d={accuracy2:.2f}%"
                if self.verbose_loss:
                    log_str += f"; Lcls:{loss_cs.item():.4f}; Lcoop:{loss_coop.item():.4f}; Liid:{loss_iid.item():.4f}; Lmme:{loss_mme.item():.4f}"
                
                log_file.write(log_str + "\n")
                log_file.flush()
                print(log_str)
                
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
        
        print(f"目标域适应完成 (CoOp Multi-Template): {task_name}! 最佳准确率: {best_acc:.2f}%")
        return output_dir
