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

"""
基于 RobustSpeech 框架实现的 Whisper 通用语言对抗攻击 (Universal Language Attack)。

本脚本依赖 RobustSpeech (rs) 提供的三大核心基础设施：
1. 攻击基类 (Inheritance): 继承 TrainableAttacker 和 ASRLinfPGDAttack，复用了标准的 PGD 攻击循环、参数检查点保存 (Checkpointer) 及实验生命周期管理，避免重复造轮子。
2. 约束工具 (Utilities): 使用 linf_clamp 等数学工具函数，确保生成的通用扰动严格满足 L-inf 范数约束，无需手动处理复杂的梯度投影。
3. 模型绑定 (Binding): 作为中间层无缝对接 SpeechBrain 的 WhisperASR 接口，实现了从攻击代码到目标模型的端到端前向传播与 Loss 计算。
"""


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
    def __init__(
        self,
        asr_brain,
        *args,
        language="es",
        targeted_for_language=True,
        nb_epochs=10,
        eps_item=0.001,
        success_every=10,
        univ_perturb=None,
        epoch_counter=None,
        # ======== 新增：日志/进度条相关配置（不影响攻击逻辑）========
        log_every=20,          # 每隔多少个 batch 输出一次训练统计
        show_pgd_pbar=False,   # 是否显示 PGD 内循环进度条（会比较慢）
        ema_alpha=0.05,        # loss 的滑动平均系数
        **kwargs
    ):
        # 1. 构造目标语言 token
        # 如果传入 "es"，处理成 "<|es|>"，这是 Whisper 特有的 token 格式
        self.language = "<|" + language.strip("<|>") + "|>"

        # 将 token 文本编码为对应的整数 ID，并放到设备上
        # self.lang_token = torch.LongTensor(asr_brain.tokenizer.encode(self.language)).to(asr_brain.device)
        # 修改为：
        self.lang_token = torch.LongTensor(
            asr_brain.tokenizer.encode(self.language, allowed_special="all")
        ).to(asr_brain.device)

        # 2. 初始化父类 ASRLinfPGDAttack
        # WhisperLangID 是一个包装类，用于计算语言识别的 Loss
        # targeted=True 表示这是有目标攻击（一定要让模型输出特定结果，而不是只要出错就行）
        ASRLinfPGDAttack.__init__(
            self,
            WhisperLangID(asr_brain, self.lang_token),
            *args,
            targeted=targeted_for_language,
            **kwargs
        )

        # 3. 初始化通用扰动变量
        self.univ_perturb = univ_perturb
        if self.univ_perturb is None:
            # 如果未提供，则初始化一个长度为 MAXLEN (30s) 的可训练参数张量
            # TensorModule 是一个简单的封装，让 tensor 可以被 saved/loaded
            self.univ_perturb = rs.adversarial.utils.TensorModule(size=(MAXLEN,))

        # 4. 训练超参数设置
        self.nb_epochs = nb_epochs          # 总训练轮数（注意：原实现用 epoch_counter 控制迭代轮数）
        self.eps_item = eps_item            # 每次迭代更新扰动的步长（L_inf 约束下的步长）
        self.success_every = success_every  # 每隔多少轮在整个数据集上评估一次成功率

        self.epoch_counter = epoch_counter
        if self.epoch_counter is None:
            self.epoch_counter = range(100)  # 默认迭代器，最大 100 轮

        # ======== 新增：训练过程记录（方便你画 loss 曲线）========
        self.log_every = log_every
        self.show_pgd_pbar = show_pgd_pbar
        self.ema_alpha = ema_alpha

        self.train_history = []  # 逐 batch 记录关键指标
        self.eval_history = []   # 逐 eval 记录 success/loss
        self._warned_lang_shape = False

    def fit(self, loader):
        """
        训练入口函数：调用内部方法开始计算通用扰动
        """
        return self._compute_universal_perturbation(loader)

    def _safe_scalar(self, x, reduce="mean"):
        """
        新增工具函数：把标量/向量 loss 转为 python float，避免 .item() 报错
        """
        if torch.is_tensor(x):
            if x.numel() == 1:
                return x.item()
            if reduce == "max":
                return x.max().item()
            return x.mean().item()
        return float(x)

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

        # 新增：滑动平均 loss（用于进度条稳定显示）
        ema_total_loss = None
        ema_lang_loss = None

        # --- 外层循环：Epoch 迭代 ---
        epoch_iter = tqdm(
            self.epoch_counter,
            desc="Epoch",
            dynamic_ncols=True
        )

        for epoch in epoch_iter:

            # --- 内层循环：遍历数据 Batch ---
            batch_iter = tqdm(
                enumerate(loader),
                total=len(loader) if hasattr(loader, "__len__") else None,
                desc=f"Train (epoch={epoch})",
                dynamic_ncols=True,
                leave=False
            )

            for idx, batch in batch_iter:
                batch = batch.to(self.asr_brain.device)
                wav_init, wav_lens = batch.sig  # 获取原始语音波形和长度

                # --- 1. 对齐扰动与输入语音的长度 ---
                # 逻辑：扰动 delta 是固定的 30s 长，但输入语音 wav_init 长度不一
                if wav_init.shape[1] <= delta.shape[0]:
                    # 情况 A：语音短于 30s
                    # 随机选择扰动的一个切片起始点，截取一段加到语音上（增加鲁棒性）
                    begin = torch.randint(delta.shape[0] - wav_init.shape[1] - 1, size=(1,))
                    delta_x = delta[begin: begin + wav_init.shape[1]].detach()
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
                r.requires_grad_()  # 开启梯度记录，以便对 r 求导

                # 新增：记录本 batch 实际跑了多少步 PGD（用于观测早停）
                used_pgd_steps = 0
                loss = None
                lang_loss = None
                l2_norm = None
                used_pgd_steps = 0

                # --- 3. PGD 攻击迭代优化 (nb_iter 次) ---
                pgd_range = range(self.nb_iter)
                if self.show_pgd_pbar:
                    pgd_range = tqdm(pgd_range, desc="PGD", dynamic_ncols=True, leave=False)
                # 新增：确保 batch 级日志一定有值

                for i in pgd_range:
                    used_pgd_steps = i + 1
                    r_batch = r.unsqueeze(0).expand(delta_batch.size())

                    # 前向传播：输入 = 原始语音 + 通用扰动 + 当前微调量 r
                    batch.sig = wav_init + delta_batch + r_batch, wav_lens

                    # 获取模型预测
                    predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)

                    # 计算 Loss：目标是让预测结果接近目标语言 token
                    lang_loss = self.asr_brain.compute_objectives(predictions, batch, rs.Stage.ATTACK)

                    # 辅助 Loss：L2 正则化，希望 r 越小越好（微调量不要太大）
                    l2_norm = r.norm(p=2).to(self.asr_brain.device)

                    # 总 Loss
                    loss = 0.5 * l2_norm + lang_loss

                    # 提前停止条件：如果 Loss 已经很小，说明攻击成功，跳出循环
                    # 注意：lang_loss 可能是向量，这里沿用你原逻辑用 max()
                    if torch.is_tensor(lang_loss):
                        if lang_loss.max() < 0.1:
                            break
                    else:
                        if float(lang_loss) < 0.1:
                            break


                    loss.backward()  # 反向传播计算梯度

                    # --- 4. 梯度更新 (PGD 核心公式) ---
                    grad_sign = r.grad.data.sign()  # 取梯度符号（L_inf 攻击特征）

                    # 更新 r：沿着梯度下降方向走（让 loss 变小？注意这里是 Targeted Attack）
                    # 这里的代码写的是 `r - ...`，通常 Targeted Attack 也是最小化 Target Loss。
                    r.data = r.data - self.rel_eps_iter * self.eps_item * grad_sign

                    # 截断 r：保证 r 自身不要太大
                    r.data = linf_clamp(r.data, self.eps_item)

                    # 截断总扰动：保证 (通用扰动 + 微调量) 总幅度不超过 eps 阈值
                    r.data = linf_clamp(delta_x + r.data, self.eps) - delta_x

                    # 清空梯度，准备下一次迭代
                    if r.grad is not None:
                        r.grad.data.zero_()
                if loss is None:
                    with torch.no_grad():
                        r_batch = r.unsqueeze(0).expand(delta_batch.size())
                        batch.sig = wav_init + delta_batch + r_batch, wav_lens
                        predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)
                        lang_loss = self.asr_brain.compute_objectives(predictions, batch, rs.Stage.ATTACK)
                        l2_norm = r.norm(p=2).to(self.asr_brain.device)
                        loss = 0.5 * l2_norm + lang_loss
                # 确保最终的 delta_x 符合约束
                delta_x = linf_clamp(delta_x + r.data, self.eps)

                # --- 5. 将更新后的 delta_x 写回全局通用扰动 delta ---
                # 这是一个“累积更新”的过程：每个 batch 都会微调全局扰动的一部分
                if delta.shape[0] <= delta_x.shape[0]:
                    delta = delta_x[:delta.shape[0]].detach()
                else:
                    # 对应之前的随机切片，只更新被选中的那一段
                    delta[begin: begin + wav_init.shape[1]] = delta_x.detach()

                # ======== 新增：训练统计（loss、扰动幅度、pgd steps）========
                with torch.no_grad():
                    lang_mean = self._safe_scalar(lang_loss, reduce="mean")
                    lang_max = self._safe_scalar(lang_loss, reduce="max")
                    l2_val = l2_norm.item() if torch.is_tensor(l2_norm) else float(l2_norm)

                    # 注意：loss 可能是张量（向量），这里也做 safe reduce
                    total_mean = self._safe_scalar(loss, reduce="mean")

                    r_linf = r.data.abs().max().item()
                    delta_linf = delta.abs().max().item()

                    # 滑动平均（用于进度条显示更平滑）
                    if ema_total_loss is None:
                        ema_total_loss = total_mean
                        ema_lang_loss = lang_mean
                    else:
                        ema_total_loss = (1 - self.ema_alpha) * ema_total_loss + self.ema_alpha * total_mean
                        ema_lang_loss = (1 - self.ema_alpha) * ema_lang_loss + self.ema_alpha * lang_mean

                    # 存历史（你后续画曲线会非常方便）
                    self.train_history.append({
                        "epoch": int(epoch),
                        "batch": int(idx),
                        "pgd_steps": int(used_pgd_steps),
                        "lang_loss_mean": float(lang_mean),
                        "lang_loss_max": float(lang_max),
                        "l2_norm": float(l2_val),
                        "total_loss_mean": float(total_mean),
                        "r_linf": float(r_linf),
                        "delta_linf": float(delta_linf),
                    })

                    # 更新进度条 postfix（每个 batch 都更新，但显示成本很低）
                    batch_iter.set_postfix({
                        "ema_total": f"{ema_total_loss:.4f}",
                        "ema_lang": f"{ema_lang_loss:.4f}",
                        "pgd": used_pgd_steps,
                        "δ∞": f"{delta_linf:.4f}",
                        "r∞": f"{r_linf:.4f}",
                    })

                    # 可选：每隔 log_every 个 batch 再额外打印一行更完整的日志
                    if (self.log_every is not None) and (self.log_every > 0) and (idx % self.log_every == 0):
                        print(
                            f"[train] epoch={epoch} batch={idx} "
                            f"pgd_steps={used_pgd_steps} "
                            f"lang(mean/max)={lang_mean:.4f}/{lang_max:.4f} "
                            f"total(mean)={total_mean:.4f} "
                            f"l2={l2_val:.4f} "
                            f"delta_linf={delta_linf:.6f} r_linf={r_linf:.6f}"
                        )

            # --- 评估阶段：每隔 success_every 轮检查一次成功率 ---
            if (epoch % self.success_every) == 0:
                print(f'Check success rate after epoch {epoch}')
                total_sample = 0.0
                fooled_sample = 0.0
                eval_loss_sum = 0.0

                # 目标 token 的“标量形式”（如果你的 lang_token 不是单 token，这里至少不会直接崩）
                target_tok = self.lang_token
                if torch.is_tensor(target_tok) and target_tok.numel() >= 1:
                    target_tok_scalar = target_tok.view(-1)[0]
                else:
                    target_tok_scalar = target_tok

                # 遍历整个 loader 进行验证
                eval_iter = tqdm(loader, dynamic_ncols=True, desc=f"Eval (epoch={epoch})", leave=False)
                for idx_eval, batch in enumerate(eval_iter):
                    batch = batch.to(self.asr_brain.device)
                    wav_init, wav_lens = batch.sig

                    # 再次构造扰动并叠加（不求导，只推理）
                    delta_x = torch.zeros_like(wav_init[0])
                    if wav_init.shape[1] <= delta.shape[0]:
                        begin = torch.randint(delta.shape[0] - wav_init.shape[1] - 1, size=(1,))
                        delta_x = delta[begin: begin + wav_init.shape[1]]
                    else:
                        delta_x[:delta.shape[0]] = delta

                    delta_batch = delta_x.unsqueeze(0).expand(wav_init.size()).to(self.asr_brain.device)
                    batch.sig = wav_init + delta_batch, wav_lens

                    # 获取预测结果
                    predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)

                    # 你原来假设 predictions 返回 (token, ..., ...)
                    language_tokens_pred, _, _ = predictions

                    # 统计：如果预测 token 等于目标 lang_token，则视为“被愚弄成功”
                    # 这里做一个尽量宽容的处理：优先按“每个样本一个 token”的情况统计
                    total_sample += float(batch.batchsize)

                    if torch.is_tensor(language_tokens_pred):
                        # 常见情况：language_tokens_pred shape = (B,) 或 (B,1)
                        pred_flat = language_tokens_pred.view(-1)

                        # 如果 target_tok 不是单 token，严格相等可能维度对不上，这里至少保证不崩
                        if torch.is_tensor(target_tok_scalar):
                            cmp = (pred_flat == target_tok_scalar)
                        else:
                            cmp = (pred_flat == torch.tensor(target_tok_scalar, device=pred_flat.device))

                        fooled_sample += float(cmp.sum().item())
                    else:
                        # 极端情况：predictions 不是 tensor
                        if not self._warned_lang_shape:
                            print("[warn] language_tokens_pred is not a tensor; success stats may be unreliable.")
                            self._warned_lang_shape = True

                    # loss 统计
                    ll = self.asr_brain.compute_objectives(predictions, batch, rs.Stage.ATTACK)
                    eval_loss_sum += self._safe_scalar(ll, reduce="mean")

                    # 更新 eval 进度条
                    cur_sr = (fooled_sample / total_sample) if total_sample > 0 else 0.0
                    cur_loss = eval_loss_sum / (idx_eval + 1)
                    eval_iter.set_postfix({
                        "sr": f"{cur_sr:.3f}",
                        "loss": f"{cur_loss:.4f}",
                    })

                success_rate = fooled_sample / total_sample if total_sample > 0 else 0.0
                mean_eval_loss = eval_loss_sum / (idx_eval + 1) if (idx_eval + 1) > 0 else float("nan")

                print(f'SUCCESS RATE IS {success_rate:.4f}')
                print(f'LOSS IS {mean_eval_loss:.4f}')

                # 新增：记录 eval 历史
                self.eval_history.append({
                    "epoch": int(epoch),
                    "success_rate": float(success_rate),
                    "mean_loss": float(mean_eval_loss),
                    "fooled": float(fooled_sample),
                    "total": float(total_sample),
                    "delta_linf": float(delta.abs().max().item()),
                })

                # 保存最佳扰动
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    self.univ_perturb.tensor.data = delta.detach()
                    self.checkpointer.save_and_keep_only()  # 保存 checkpoint

            # 更新 epoch 进度条信息
            epoch_iter.set_postfix({
                "best_sr": f"{best_success_rate:.3f}",
                "last_sr": f"{success_rate:.3f}",
                "δ∞": f"{delta.abs().max().item():.4f}",
            })

        # 注意：best_success_rate 是比例（0~1），这里不要打印成百分号误导
        print(f"Training finisihed. Best success rate: {best_success_rate:.4f}")

        # 把最终 delta 写回（即使不是 best，也能保留训练末状态）
        self.univ_perturb.tensor.data = delta.detach()

        return self.univ_perturb

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
        save_input = batch.sig[0]  # 保存原始输入
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
