from robust_speech.adversarial.attacks.pgd import SNRPGDAttack, ASRLinfPGDAttack  # 引入 PGD 攻击基类，用于后续继承

from sb_whisper_binding import WhisperASR

from whisper_with_gradients import detect_language_with_gradients
from whisper.audio import CHUNK_LENGTH, SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram
import robust_speech as rs
import torch.nn as nn
import torch
from robust_speech.adversarial.utils import TargetGenerator
import copy 
from robust_speech.adversarial.attacks.attacker import TrainableAttacker
from lang_attack import WhisperLangID
from robust_speech.adversarial.utils import (
    l2_clamp_or_normalize,
    linf_clamp,
    rand_assign,
)
from tqdm import tqdm

# 定义对抗扰动的最大长度：30秒 * 16000Hz采样率 = 480,000 个采样点
# Whisper 处理音频的窗口通常是 30 秒
MAXLEN = 16000 * 30  

class UniversalWhisperLanguageAttack(TrainableAttacker, ASRLinfPGDAttack):
    """
    通用语言攻击类：
    目标是学习一个单一的、可迁移的扰动（Universal Perturbation），
    使得该扰动加到任意语音上都能促使 Whisper 输出指定的目标语言 token。
    继承自 TrainableAttacker (表明需要训练/拟合) 和 ASRLinfPGDAttack (表明基于 Linf 范数的 PGD 攻击)。
    """
    def __init__(self, asr_brain, *args, language="es", targeted_for_language=True, nb_epochs=10, eps_item=0.001, success_every=10, univ_perturb=None, epoch_counter=None, **kwargs):
        # 1. 构造目标语言 token
        # 如果传入 "es"，处理成 "<|es|>"，这是 Whisper 特有的 token 格式
        self.language = "<|" + language.strip("<|>") + "|>"  
        
        # 将 token 文本编码为对应的整数 ID，并放到设备上
        self.lang_token = torch.LongTensor(asr_brain.tokenizer.encode(self.language)).to(asr_brain.device)
        
        # 2. 初始化父类 ASRLinfPGDAttack
        # WhisperLangID 是一个包装类，用于计算语言识别的 Loss
        # targeted=True 表示这是有目标攻击（一定要让模型输出特定结果，而不是只要出错就行）
        ASRLinfPGDAttack.__init__(self, WhisperLangID(asr_brain, self.lang_token), *args, targeted=targeted_for_language, **kwargs)
        
        # 3. 初始化通用扰动变量
        self.univ_perturb = univ_perturb
        if self.univ_perturb is None:
            # 如果未提供，则初始化一个长度为 MAXLEN (30s) 的可训练参数张量
            # TensorModule 是一个简单的封装，让 tensor 可以被 saved/loaded
            self.univ_perturb = rs.adversarial.utils.TensorModule(size=(MAXLEN,))
        
        # 4. 训练超参数设置
        self.nb_epochs = nb_epochs          # 总训练轮数
        self.eps_item = eps_item            # 每次迭代更新扰动的步长（L_inf 约束下的步长）
        self.success_every = success_every  # 每隔多少轮在整个数据集上评估一次成功率
        
        self.epoch_counter = epoch_counter
        if self.epoch_counter is None:
            self.epoch_counter = range(100) # 默认迭代器，最大 100 轮

    def fit(self, loader):
        """
        训练入口函数：调用内部方法开始计算通用扰动
        """
        return self._compute_universal_perturbation(loader)

    def _compute_universal_perturbation(self, loader):
        """
        核心训练逻辑：通过在数据集上多轮迭代，优化通用扰动 delta
        """
        # 将扰动张量移动到 GPU/CPU
        self.univ_perturb = self.univ_perturb.to(self.asr_brain.device)
        
        # 根据配置决定模型是否开启训练模式（通常攻击时模型参数是固定的，设为 eval）
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        else:
            self.asr_brain.module_eval()

        # 获取当前的扰动数据（delta），初始通常为全0或随机
        delta = self.univ_perturb.tensor.data
        success_rate = 0
        best_success_rate = -100

        # --- 外层循环：Epoch 迭代 ---
        for epoch in self.epoch_counter:
            
            # --- 内层循环：遍历数据 Batch ---
            for idx, batch in enumerate(loader):
                batch = batch.to(self.asr_brain.device)
                wav_init, wav_lens = batch.sig # 获取原始语音波形和长度
                
                # --- 1. 对齐扰动与输入语音的长度 ---
                # 逻辑：扰动 delta 是固定的 30s 长，但输入语音 wav_init 长度不一
                if wav_init.shape[1] <= delta.shape[0]:
                    # 情况 A：语音短于 30s
                    # 随机选择扰动的一个切片起始点，截取一段加到语音上（增加鲁棒性）
                    begin = torch.randint(delta.shape[0] - wav_init.shape[1] - 1, size=(1,))
                    delta_x = delta[begin : begin + wav_init.shape[1]].detach()
                else:
                    # 情况 B：语音长于 30s（较少见，因 MAXLEN 已设为 30s）
                    # 补零扩展扰动以匹配语音长度
                    delta_x = torch.zeros_like(wav_init[0])
                    delta_x[: delta.shape[0]] = delta.detach()
                
                # 将扰动扩展到 Batch 维度
                delta_batch = delta_x.unsqueeze(0).expand(wav_init.size())

                # --- 2. 初始化局部更新量 r ---
                # 我们不是直接修改 delta，而是计算一个临时的微小增量 r
                r = torch.zeros_like(delta_x)
                r.requires_grad_() # 开启梯度记录，以便对 r 求导

                # --- 3. PGD 攻击迭代优化 (nb_iter 次) ---
                for i in range(self.nb_iter):
                    r_batch = r.unsqueeze(0).expand(delta_batch.size())

                    # 前向传播：输入 = 原始语音 + 通用扰动 + 当前微调量 r
                    batch.sig = wav_init + delta_batch + r_batch, wav_lens
                    
                    # 获取模型预测
                    predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)
                    
                    # 计算 Loss：目标是让预测结果接近目标语言 token
                    lang_loss = self.asr_brain.compute_objectives(predictions, batch, rs.Stage.ATTACK)
                    
                    # 辅助 Loss：L2 正则化，希望 r 越小越好（微调量不要太大）
                    l2_norm = r.norm(p=2).to(self.asr_brain.device)

                    # 提前停止条件：如果 Loss 已经很小，说明攻击成功，跳出循环
                    if lang_loss.max() < 0.1:
                        break
                    
                    # 总 Loss
                    loss = 0.5 * l2_norm + lang_loss
                    loss.backward() # 反向传播计算梯度

                    # --- 4. 梯度更新 (PGD 核心公式) ---
                    grad_sign = r.grad.data.sign() # 取梯度符号（L_inf 攻击特征）
                    
                    # 更新 r：沿着梯度下降方向走（让 loss 变小？注意这里是 Targeted Attack）
                    # 这里的代码写的是 `r - ...`，通常 Targeted Attack 也是最小化 Target Loss。
                    r.data = r.data - self.rel_eps_iter * self.eps_item * grad_sign
                    
                    # 截断 r：保证 r 自身不要太大
                    r.data = linf_clamp(r.data, self.eps_item)
                    
                    # 截断总扰动：保证 (通用扰动 + 微调量) 总幅度不超过 eps 阈值
                    r.data = linf_clamp(delta_x + r.data, self.eps) - delta_x

                    # 清空梯度，准备下一次迭代
                    r.grad.data.zero_()

                # 确保最终的 delta_x 符合约束
                delta_x = linf_clamp(delta_x + r.data, self.eps)

                # --- 5. 将更新后的 delta_x 写回全局通用扰动 delta ---
                # 这是一个“累积更新”的过程：每个 batch 都会微调全局扰动的一部分
                if delta.shape[0] <= delta_x.shape[0]:
                    delta = delta_x[:delta.shape[0]].detach()
                else:
                    # 对应之前的随机切片，只更新被选中的那一段
                    delta[begin : begin + wav_init.shape[1]] = delta_x.detach()

            # --- 评估阶段：每隔 success_every 轮检查一次成功率 ---
            if (epoch % self.success_every) == 0:
                print(f'Check success rate after epoch {epoch}')
                total_sample = 0.
                fooled_sample = 0.
                loss = 0

                # 遍历整个 loader 进行验证
                for idx, batch in enumerate(tqdm(loader, dynamic_ncols=True)):
                    batch = batch.to(self.asr_brain.device)
                    wav_init, wav_lens = batch.sig

                    # 再次构造扰动并叠加（不求导，只推理）
                    delta_x = torch.zeros_like(wav_init[0])
                    if wav_init.shape[1] <= delta.shape[0]:
                        begin = torch.randint(delta.shape[0] - wav_init.shape[1] - 1, size=(1,))
                        delta_x = delta[begin : begin + wav_init.shape[1]]
                    else:
                        delta_x[:delta.shape[0]] = delta
                    
                    delta_batch = delta_x.unsqueeze(0).expand(wav_init.size()).to(self.asr_brain.device)
                    batch.sig = wav_init + delta_batch, wav_lens

                    # 获取预测结果
                    predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)
                    language_tokens_pred, _, _ = predictions # 假设 predictions 返回 (token, ..., ...)

                    # 统计：如果预测 token 等于目标 lang_token，则视为“被愚弄成功”
                    total_sample += batch.batchsize
                    fooled_sample += (language_tokens_pred == self.lang_token).sum()
                    loss += self.asr_brain.compute_objectives(predictions, batch, rs.Stage.ATTACK).item()

                success_rate = fooled_sample / total_sample
                print(f'SUCCESS RATE IS {success_rate:.4f}')
                print(f'LOSS IS {(loss / (idx + 1)):.4f}')

                # 保存最佳扰动
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    self.univ_perturb.tensor.data = delta.detach()
                    self.checkpointer.save_and_keep_only() # 保存 checkpoint

        print(f"Training finisihed. Best success rate: {best_success_rate:.2f}%") 

    def perturb(self, batch):
        """
        攻击推理接口：
        给定一个 batch，应用训练好的通用扰动，返回对抗样本
        """
        # 设置模型模式
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        else:
            self.asr_brain.module_eval()

        save_device = batch.sig[0].device
        batch = batch.to(self.asr_brain.device)
        save_input = batch.sig[0] # 保存原始输入
        wav_init = torch.clone(save_input)

        # 取出训练好的通用扰动
        delta = self.univ_perturb.tensor.data.to(self.asr_brain.device)

        # 长度适配：截断或补零
        if wav_init.shape[1] <= delta.shape[0]:
            delta_x = delta[:wav_init.shape[1]]
        else:
            delta_x = torch.zeros_like(wav_init[0])
            delta[:delta.shape[0]] = delta
        
        # 叠加扰动
        delta_batch = delta_x.unsqueeze(0).expand(wav_init.size())
        wav_adv = wav_init + delta_batch

        # 恢复 batch 原始状态（避免副作用修改原数据）
        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        self.asr_brain.module_eval()

        # 返回对抗样本张量
        return wav_adv.data.to(save_device)
