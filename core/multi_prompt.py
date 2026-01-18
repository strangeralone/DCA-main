"""DCA + Multi-Prompt 训练器模块

继承 DCA+CLIP 训练器，使用多模板增强策略生成更鲁棒的文本特征。
多个模板的文本特征取平均，增强语义覆盖能力。
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


# 默认多模板列表
DEFAULT_TEMPLATES = [
    "a photo of a {}",
    "a {} in the scene",
    "an image of a {}",
    "a professional photo of a {}",
    "a picture of a {}",
    "a {} image",
    "the {} in the photo",
    "a good photo of a {}",
]


class MultiPromptTrainer(DCATrainer):
    """DCA + Multi-Prompt 训练器类
    
    使用多个文本模板生成 CLIP 文本特征，并取平均作为最终的文本表示。
    这种方式可以增强语义覆盖能力，提升域适应效果。
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        mp_config = self.method_config.get('multi_prompt', {})
        self.clip_model_name = mp_config.get('model', 'ViT-B/16')
        self.clip_loss_weight = mp_config.get('loss_weight', 1.0)
        self.clip_temperature = mp_config.get('temperature', 100)
        self.clip_entropy_ratio = mp_config.get('entropy_ratio', 0.15)
        self.clip_top_k_ratio = mp_config.get('top_k_ratio', 0.2)
        
        # 多模板配置
        self.templates = mp_config.get('templates', DEFAULT_TEMPLATES)
        
        self.clip_model = None
        self.text_features = None  # 融合后的文本特征
        
        # 调试配置
        self.debug = config.get('debug', False)
        self.verbose_loss = config.get('verbose_loss', True)
    
    def _load_clip_model(self):
        """加载并冻结 CLIP 模型"""
        print(f"加载 CLIP 模型: {self.clip_model_name}...")
        self.clip_model, _ = clip.load(self.clip_model_name, device=self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        print("CLIP 模型已加载并冻结!")
    
    def _prepare_text_features(self, class_names):
        """预计算多模板融合的 CLIP 文本特征
        
        对每个类别使用多个模板生成文本特征，然后取平均。
        """
        print(f"使用 {len(self.templates)} 个模板生成文本特征...")
        
        all_features = []
        
        for template in self.templates:
            prompts = [template.format(name.replace('_', ' ')) for name in class_names]
            text_tokens = clip.tokenize(prompts).to(self.device)
            
            with torch.no_grad():
                features = self.clip_model.encode_text(text_tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features)
        
        # 取平均 [num_templates, num_classes, dim] -> [num_classes, dim]
        stacked = torch.stack(all_features, dim=0)
        self.text_features = stacked.mean(dim=0)
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        
        print(f"多模板文本特征已准备: {self.text_features.shape}")
        print(f"使用的模板: {self.templates}")
    
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
        """将配置信息写入日志文件（包含多模板配置）"""
        super()._write_config_to_log(log_file, task_type, source_idx, target_idx)
        
        # 追加 Multi-Prompt 配置
        log_file.write("Multi-Prompt 配置:\n")
        log_file.write(f"  model: {self.clip_model_name}\n")
        log_file.write(f"  loss_weight: {self.clip_loss_weight}\n")
        log_file.write(f"  entropy_ratio: {self.clip_entropy_ratio}\n")
        log_file.write(f"  top_k_ratio: {self.clip_top_k_ratio}\n")
        log_file.write(f"  temperature: {self.clip_temperature}\n")
        log_file.write(f"  num_templates: {len(self.templates)}\n")
        log_file.write(f"  templates: {self.templates}\n")
        log_file.write(f"  debug: {self.debug}\n")
        log_file.write(f"  verbose_loss: {self.verbose_loss}\n")
        log_file.write("=" * 60 + "\n\n")
        log_file.flush()
    
    def train_target(self, source_idx, target_idx, log_file=None):
        """目标域自适应（带多模板增强）"""
        start_time = datetime.now()
        task_name = f"{self.domains[source_idx]} -> {self.domains[target_idx]}"
        print(f"目标域适应 (带多模板增强): {task_name}...")
        
        # 构建网络
        self._build_networks()
        
        # 加载 CLIP 模型
        self._load_clip_model()
        
        # 准备多模板文本特征
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
        pbar = tqdm(total=max_iter, desc=f"Target+MultiPrompt [{task_name}]", ncols=120)
        
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
            
            # Step B: 联合训练 netG + netD (带多模板蒸馏)
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
            
            # 多模板蒸馏损失 - 难样本选择
            entropy1 = -torch.sum(softmax_out1 * torch.log(softmax_out1 + 1e-8), dim=1)
            entropy2 = -torch.sum(softmax_out2 * torch.log(softmax_out2 + 1e-8), dim=1)
            avg_entropy = (entropy1 + entropy2) / 2
            
            max_entropy = np.log(self.class_num)
            entropy_threshold = max_entropy * self.clip_entropy_ratio
            
            discrepancy = torch.sum(SKL(softmax_out1, softmax_out2), dim=1)
            discrepancy_threshold = discrepancy.quantile(1 - self.clip_top_k_ratio)
            
            high_entropy_mask = avg_entropy > entropy_threshold
            high_discrepancy_mask = discrepancy > discrepancy_threshold
            difficult_mask = high_entropy_mask | high_discrepancy_mask
            topk_indices = torch.where(difficult_mask)[0]
            
            if len(topk_indices) > 0:
                clip_input = F.interpolate(inputs_test[topk_indices], size=(224, 224), mode='bicubic')
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(clip_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    clip_logits = image_features @ self.text_features.T
                    clip_probs = F.softmax(clip_logits * self.clip_temperature, dim=1)
                
                loss_mp = F.kl_div(log_sm(outputs1[topk_indices]), clip_probs.float().detach(), reduction='batchmean') + \
                          F.kl_div(log_sm(outputs2[topk_indices]), clip_probs.float().detach(), reduction='batchmean')
            else:
                loss_mp = torch.tensor(0.0).to(self.device)
            
            total_loss2 += self.clip_loss_weight * loss_mp
            
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
            
            # 定期评估
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                self.netG.eval()
                self.netF.eval()
                
                _, acc_list1, accuracy1 = cal_acc(dset_loaders["test"], self.netG, self.netF, self.netC, device=self.device)
                _, acc_list2, accuracy2 = cal_acc_easy(dset_loaders["test"], self.netG, self.netD, device=self.device)
                
                log_str = f"Iter:{iter_num}/{max_iter}; Acc_c={accuracy1:.2f}%, Acc_d={accuracy2:.2f}%"
                if self.verbose_loss:
                    log_str += f"; Lcls:{loss_cs.item():.4f}; Lmp:{loss_mp.item():.4f}; Lmme:{loss_mme.item():.4f}; Lcb:{loss_cb.item():.4f}; Lmix:{loss_mix.item():.4f}"
                else:
                    log_str += f"; Lcls:{loss_cs.item():.4f}; Lmp:{loss_mp.item():.4f}"
                
                log_file.write(log_str + "\n")
                log_file.flush()
                
                pbar.set_postfix({'Acc_c': f'{accuracy1:.2f}%', 'Acc_d': f'{accuracy2:.2f}%'})
                
                # 早停检查
                if accuracy1 > best_acc:
                    best_acc = accuracy1
                    best_iter = iter_num
                    no_improve_count = 0
                    if self.config.get('savemodel', True):
                        torch.save(self.netG.state_dict(), osp.join(output_dir, "target_G.pt"))
                        torch.save(self.netF.state_dict(), osp.join(output_dir, "target_F.pt"))
                        torch.save(self.netC.state_dict(), osp.join(output_dir, "target_C.pt"))
                        torch.save(self.netD.state_dict(), osp.join(output_dir, "target_D.pt"))
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
        
        # 写入结束时间
        self._write_end_time(log_file, start_time)
        
        print(f"目标域适应完成 (多模板增强): {task_name}! 最佳准确率: {best_acc:.2f}%")
        return output_dir
