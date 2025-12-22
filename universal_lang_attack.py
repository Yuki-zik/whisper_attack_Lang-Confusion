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
from collections import Counter
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
        success_every=20,
        univ_perturb=None,
        epoch_counter=None,
        # ======== 新增：日志/进度条相关配置（不影响攻击逻辑）========
        log_every=20,          # 每隔多少个 batch 输出一次训练统计
        log_lang_pred_every=20,  # 每隔多少个 epoch 打印一次语种预测结果，便于观察 delta 的攻击倾向
        log_lang_pred_samples=5,  # 每次打印时输出多少条样本的预测，便于观察多条样本的偏移
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

        # tokenizer 中的语言 token -> 语言码映射，用于日志解码
        self.lang_token_to_code = {}
        if hasattr(asr_brain, "tokenizer"):
            toks = getattr(asr_brain.tokenizer, "all_language_tokens", None)
            codes = getattr(asr_brain.tokenizer, "all_language_codes", None)
            if toks is not None and codes is not None and len(toks) == len(codes):
                self.lang_token_to_code = {int(t): c for t, c in zip(toks, codes)}

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
        self.log_lang_pred_every = log_lang_pred_every
        self.log_lang_pred_samples = log_lang_pred_samples
        self.show_pgd_pbar = show_pgd_pbar
        self.ema_alpha = ema_alpha
        # 语言识别前向的分块大小；大 batch + 多步 PGD 时可按此切分 micro-batch 逐块反传，降低显存峰值
        self.lang_microbatch_size = getattr(asr_brain.hparams, "lang_microbatch_size", None)

        # 尝试记录数据集的“源语言”标注，便于日志对照；不存在则留空
        src_lang = getattr(asr_brain.hparams, "lang_CV", None)
        if src_lang is None:
            src_lang = getattr(asr_brain.hparams, "language", None)
        if src_lang:
            src_lang = str(src_lang).strip("<|>")
            self.source_language = f"<|{src_lang}|>({src_lang})"
        else:
            self.source_language = None

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

    def _decode_lang_token(self, tok):
        """尝试将语言 token id 反解为可读文本，失败则返回原值字符串。"""
        if torch.is_tensor(tok):
            tok = tok.detach()
            if tok.numel() == 1:
                tok = tok.item()
            else:
                tok = tok.view(-1).tolist()
        try:
            # Whisper tokenizer 对语言 token 的处理有时不经过标准 decode，这里先尝试标准 decode，
            # 失败后回退到 tokenizer.tokenizer.decode_single_token_bytes 的底层接口，确保语言 ID 能映射到 <|xx|> 文本。
            decoded = None
            tid = None
            if isinstance(tok, (list, tuple)):
                decoded = self.asr_brain.tokenizer.decode(tok, skip_special_tokens=False)
                if tok:
                    tid = int(tok[0])
            else:
                tid = int(tok)
                decoded = self.asr_brain.tokenizer.decode([tid], skip_special_tokens=False)

            if (not decoded) and (tid is not None) and hasattr(self.asr_brain.tokenizer, "tokenizer"):
                base_tok = self.asr_brain.tokenizer.tokenizer
                if hasattr(base_tok, "decode_single_token_bytes"):
                    decoded = base_tok.decode_single_token_bytes(int(tid)).decode("utf-8")

            # 若是 <|xx|> 形式的语言 token，额外输出简洁语言码，便于快速辨识
            if isinstance(decoded, str) and decoded.startswith("<|") and decoded.endswith("|>"):
                lang_code = decoded[2:-2]
                return f"{decoded}({lang_code})"
            return decoded if decoded is not None else str(tok)
        except Exception:
            return str(tok)

    def _log_language_predictions(self, loader, delta, epoch):
        """
        额外的诊断日志：抽取一个 batch，在当前通用扰动下前向，打印语言预测。默认每 20 个 epoch 打印一次。
        不参与训练梯度，主要用于观察扰动是否把预测推向目标语言，并同时给出未加扰动的“源语言预测”。
        """
        with torch.no_grad():
            collected = 0
            adv_pred_texts = []
            clean_pred_texts = []

            for sample_batch in loader:
                sample_batch = sample_batch.to(self.asr_brain.device)
                wav_init, wav_lens = sample_batch.sig

                # 先记录未加扰动的语言预测，便于对比“源语言→扰动后的语言”
                clean_predictions = self.asr_brain.compute_forward(sample_batch, rs.Stage.ATTACK)
                clean_lang_tokens, _, _ = clean_predictions

                # 构造扰动后的对抗样本
                delta_x = torch.zeros_like(wav_init[0])
                if wav_init.shape[1] <= delta.shape[0]:
                    begin = torch.randint(delta.shape[0] - wav_init.shape[1] - 1, size=(1,))
                    delta_x = delta[begin: begin + wav_init.shape[1]]
                else:
                    delta_x[:delta.shape[0]] = delta

                delta_batch = delta_x.unsqueeze(0).expand(wav_init.size()).to(self.asr_brain.device)
                sample_batch.sig = wav_init + delta_batch, wav_lens

                predictions = self.asr_brain.compute_forward(sample_batch, rs.Stage.ATTACK)
                language_tokens_pred, _, _ = predictions

                def _collect_preds(tensor_preds, bucket, remaining):
                    added = 0
                    if torch.is_tensor(tensor_preds):
                        flat_pred = tensor_preds.view(-1)
                        for tok in flat_pred:
                            bucket.append(self._decode_lang_token(tok))
                            added += 1
                            if added >= remaining:
                                break
                    else:
                        bucket.append(f"非张量预测: {tensor_preds}")
                        added += 1
                    return added

                collected += _collect_preds(language_tokens_pred, adv_pred_texts, self.log_lang_pred_samples - collected)
                _collect_preds(clean_lang_tokens, clean_pred_texts, self.log_lang_pred_samples)

                if collected >= self.log_lang_pred_samples:
                    break

            if collected == 0:
                print("[lang-pred] 数据加载器为空，跳过语言预测打印")
                return

            target_tok_scalar = self.lang_token.view(-1)[0] if torch.is_tensor(self.lang_token) else self.lang_token
            target_text = self._decode_lang_token(target_tok_scalar)

            print(
                f"[lang-pred] epoch={epoch} target={target_text} "
                f"clean(前{len(clean_pred_texts)}条)={clean_pred_texts} "
                f"adv(前{len(adv_pred_texts)}条)={adv_pred_texts}"
            )

    def _summarize_language_shift(self, loader, delta, desc="test"):
        """
        统一的评估逻辑：在给定数据集上统计成功率、loss 以及语种变化比例。
        训练阶段（success_every）和测试阶段都复用该函数，避免重复代码。
        """
        print(f"[lang-eval] 开始在 {desc} 数据集统计语种变化和成功率")
        total_sample = 0.0
        fooled_sample = 0.0
        eval_loss_sum = 0.0
        lang_eval_total = 0.0
        lang_changed = 0.0
        lang_transition_counter = Counter()

        target_tok = self.lang_token
        if torch.is_tensor(target_tok) and target_tok.numel() >= 1:
            target_tok_scalar = target_tok.view(-1)[0]
        else:
            target_tok_scalar = target_tok

        with torch.no_grad():
            eval_iter = tqdm(
                loader, dynamic_ncols=True, desc=f"Eval ({desc})", leave=False
            )
            for idx_eval, batch in enumerate(eval_iter):
                batch = batch.to(self.asr_brain.device)
                wav_init, wav_lens = batch.sig

                delta_x = torch.zeros_like(wav_init[0])
                if wav_init.shape[1] <= delta.shape[0]:
                    begin = torch.randint(delta.shape[0] - wav_init.shape[1] - 1, size=(1,))
                    delta_x = delta[begin: begin + wav_init.shape[1]]
                else:
                    delta_x[:delta.shape[0]] = delta

                delta_batch = delta_x.unsqueeze(0).expand(wav_init.size()).to(
                    self.asr_brain.device
                )

                batch.sig = wav_init, wav_lens
                clean_predictions = self.asr_brain.compute_forward(
                    batch, rs.Stage.ATTACK
                )
                clean_lang_tokens, _, _ = clean_predictions

                batch.sig = wav_init + delta_batch, wav_lens
                predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)
                language_tokens_pred, _, _ = predictions

                total_sample += float(batch.batchsize)

                if torch.is_tensor(language_tokens_pred):
                    pred_flat = language_tokens_pred.view(-1)

                    if torch.is_tensor(clean_lang_tokens):
                        clean_flat = clean_lang_tokens.view(-1)
                        pair_len = min(clean_flat.numel(), pred_flat.numel())
                        clean_flat = clean_flat[:pair_len]
                        pred_flat = pred_flat[:pair_len]
                        diff_mask = clean_flat != pred_flat
                        lang_changed += float(diff_mask.sum().item())
                        lang_eval_total += float(pair_len)

                        for c_tok, a_tok in zip(
                            clean_flat[diff_mask], pred_flat[diff_mask]
                        ):
                            lang_transition_counter[(int(c_tok), int(a_tok))] += 1

                    if torch.is_tensor(target_tok_scalar):
                        cmp = pred_flat == target_tok_scalar
                    else:
                        cmp = pred_flat == torch.tensor(
                            target_tok_scalar, device=pred_flat.device
                        )

                    fooled_sample += float(cmp.sum().item())
                else:
                    if not self._warned_lang_shape:
                        print(
                            "[warn] language_tokens_pred is not a tensor; success stats may be unreliable."
                        )
                        self._warned_lang_shape = True

                ll = self.asr_brain.compute_objectives(
                    predictions, batch, rs.Stage.ATTACK
                )
                eval_loss_sum += self._safe_scalar(ll, reduce="mean")

                cur_sr = (fooled_sample / total_sample) if total_sample > 0 else 0.0
                cur_loss = eval_loss_sum / (idx_eval + 1)
                eval_iter.set_postfix({"sr": f"{cur_sr:.3f}", "loss": f"{cur_loss:.4f}"})

        success_rate = fooled_sample / total_sample if total_sample > 0 else 0.0
        mean_eval_loss = (
            eval_loss_sum / (idx_eval + 1) if (idx_eval + 1) > 0 else float("nan")
        )
        lang_change_rate = (
            lang_changed / lang_eval_total if lang_eval_total > 0 else 0.0
        )

        print(f'SUCCESS RATE IS {success_rate:.4f}')
        print(f'LOSS IS {mean_eval_loss:.4f}')
        print(
            f'LANGUAGE CHANGE RATE IS {lang_change_rate:.4f} (changed {lang_changed:.0f}/{lang_eval_total:.0f})'
        )

        if lang_transition_counter:
            top_trans = lang_transition_counter.most_common(5)
            decoded_trans = []
            for (src_tok, tgt_tok), cnt in top_trans:
                decoded_trans.append(
                    f"{self._decode_lang_token(src_tok)} -> {self._decode_lang_token(tgt_tok)}: {cnt}"
                )
            print(f'[lang-shift top5] {decoded_trans}')

        return {
            "success_rate": success_rate,
            "mean_loss": mean_eval_loss,
            "fooled": fooled_sample,
            "total": total_sample,
            "lang_change_rate": lang_change_rate,
            "lang_changed": lang_changed,
            "lang_eval_total": lang_eval_total,
        }

    def log_test_language_shift(self, dataset, dataloader_opts=None, split_name="TEST"):
        """
        在测试集上额外打印语种变化与成功率。
        仅用于攻击训练结束后的 eval，避免用户在 test 日志中看不到语种信息。
        """
        if dataloader_opts is None:
            dataloader_opts = {}

        loader = self.asr_brain.make_dataloader(dataset, stage=rs.Stage.ATTACK, **dataloader_opts)
        delta = self.univ_perturb.tensor.data.to(self.asr_brain.device)
        self.asr_brain.module_eval()
        self._summarize_language_shift(loader, delta, desc=split_name)

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
        # 说明：self.epoch_counter 由命令行 --epochs（或 YAML 默认）创建，控制训练的
        # 外层轮数；命令行修改 epochs，等价于改变 epoch_iter 的总迭代次数。

        for epoch in epoch_iter:

            # --- 内层循环：遍历数据 Batch ---
            batch_iter = tqdm(
                enumerate(loader),
                total=len(loader) if hasattr(loader, "__len__") else None,
                desc=f"Train (epoch={epoch})",
                dynamic_ncols=True,
                leave=False
            )
            # 说明：batch_iter 的批次数量由 DataLoader 决定；DataLoader 的 batch_size
            # 直接取自命令行参数 --batch_size（或 YAML 中同名设置）。因此命令行改动
            # batch_size 时，这里的迭代次数也会随之变化。

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
                # 说明：self.nb_iter 对应命令行 --nb_iter（或 YAML 默认值），代表每个
                # batch 中 PGD 的迭代步数；命令行增大/减小该值会直接改变此循环长度。

                for i in pgd_range:
                    used_pgd_steps = i + 1
                    r_batch = r.unsqueeze(0).expand(delta_batch.size())

                    # 前向传播：输入 = 原始语音 + 通用扰动 + 当前微调量 r
                    # 为避免一次性在大 batch 上构图撑爆显存，这里支持按 micro-batch 分块反传。
                    # 注意：lang_microbatch_size 仅影响显存占用，不会改变梯度期望，因为每块 loss
                    # 会按样本占比归一化后再累积。
                    total_batch = wav_init.size(0)
                    micro_bs = self.lang_microbatch_size or total_batch
                    micro_bs = max(1, micro_bs)

                    # 备份原始字段，便于循环内临时切片后恢复
                    orig_sig = batch.sig
                    orig_tokens = batch.tokens
                    orig_tokens_bos = batch.tokens_bos

                    lang_loss_list = []

                    for start in range(0, total_batch, micro_bs):
                        end = min(total_batch, start + micro_bs)
                        # 切片后的扰动语音
                        batch.sig = (
                            (wav_init + delta_batch + r_batch)[start:end],
                            wav_lens[start:end],
                        )
                        # 对齐文本/长度（尽管 compute_forward_lang 当前未使用 tokens，但保持完整 batch 信息）
                        batch.tokens = (
                            orig_tokens[0][start:end],
                            orig_tokens[1][start:end],
                        )
                        batch.tokens_bos = (
                            orig_tokens_bos[0][start:end],
                            orig_tokens_bos[1][start:end],
                        )

                        # 获取模型预测
                        predictions = self.asr_brain.compute_forward(
                            batch, rs.Stage.ATTACK
                        )

                        # 计算 Loss：目标是让预测结果接近目标语言 token
                        chunk_loss = self.asr_brain.compute_objectives(
                            predictions, batch, rs.Stage.ATTACK, reduction="mean"
                        )

                        # 为保持整体等价性，按样本占比缩放再累积梯度
                        chunk_scale = (end - start) / total_batch
                        (chunk_loss * chunk_scale).backward()
                        lang_loss_list.append(chunk_loss.detach())

                    # 恢复原始 batch 字段，避免影响后续逻辑
                    batch.sig = orig_sig
                    batch.tokens = orig_tokens
                    batch.tokens_bos = orig_tokens_bos

                    # 辅助 Loss：L2 正则化，希望 r 越小越好（微调量不要太大）
                    l2_norm = r.norm(p=2).to(self.asr_brain.device)
                    (0.5 * l2_norm).backward()

                    # 汇总日志用的标量（梯度已在上面分块累积完毕）
                    lang_loss = torch.stack(lang_loss_list).mean()
                    loss = 0.5 * l2_norm + lang_loss

                    # 提前停止条件：如果 Loss 已经很小，说明攻击成功，跳出循环
                    if torch.is_tensor(lang_loss):
                        if lang_loss.max() < 0.1:
                            break
                    else:
                        if float(lang_loss) < 0.1:
                            break

                    # --- 4. 梯度更新 (PGD 核心公式) ---
                    grad_sign = r.grad.data.sign()  # 取梯度符号（L_inf 攻击特征）

                    # 更新 r：沿着梯度下降方向走（让 loss 变小？注意这里是 Targeted Attack）
                    # 这里的代码写的是 `r - ...`，通常 Targeted Attack 也是最小化 Target Loss。
                    r.data = r.data - self.rel_eps_iter * self.eps_item * grad_sign
                    # 说明：self.rel_eps_iter 来源于命令行 --rel_eps_iter，self.eps_item
                    # 来源于命令行 --eps_item（若未指定则用 YAML 默认）。它们的乘积
                    # 直接控制每一步 PGD 更新的步长大小。

                    # 截断 r：保证 r 自身不要太大
                    r.data = linf_clamp(r.data, self.eps_item)

                    # 截断总扰动：保证 (通用扰动 + 微调量) 总幅度不超过 eps 阈值
                    r.data = linf_clamp(delta_x + r.data, self.eps) - delta_x
                    # 说明：self.eps 由命令行 --eps（或 YAML 默认）提供，用作 L_inf 投影
                    # 半径；命令行调大/调小 eps，会影响允许的最大扰动幅度。

                    # 清空梯度，准备下一次迭代
                    if r.grad is not None:
                        r.grad.data.zero_()
                if loss is None:
                    with torch.no_grad():
                        r_batch = r.unsqueeze(0).expand(delta_batch.size())
                        batch.sig = wav_init + delta_batch + r_batch, wav_lens
                        predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)
                        lang_loss = self.asr_brain.compute_objectives(
                            predictions, batch, rs.Stage.ATTACK, reduction="mean"
                        )
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

                    # 计算当前批次的信噪比（SNR，单位 dB），便于观察扰动强度
                    r_batch = r.unsqueeze(0).expand(delta_batch.size())
                    noise = delta_batch + r_batch
                    signal_power = wav_init.pow(2).mean()
                    noise_power = noise.pow(2).mean()
                    snr_db = float("inf")
                    if noise_power.item() > 0:
                        snr_db = (
                            10
                            * torch.log10(
                                signal_power.clamp_min(1e-12)
                                / noise_power.clamp_min(1e-12)
                            )
                        ).item()

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
                        "snr_db": float(snr_db),
                    })

                    # 更新进度条 postfix（每个 batch 都更新，但显示成本很低）
                    batch_iter.set_postfix({
                        "ema_total": f"{ema_total_loss:.4f}",
                        "ema_lang": f"{ema_lang_loss:.4f}",
                        "pgd": used_pgd_steps,
                        "δ∞": f"{delta_linf:.4f}",
                        "r∞": f"{r_linf:.4f}",
                        "SNR(dB)": f"{snr_db:.2f}",
                    })

                    # 可选：每隔 log_every 个 batch 再额外打印一行更完整的日志
                    if (self.log_every is not None) and (self.log_every > 0) and (idx % self.log_every == 0):
                        print(
                            f"[train] epoch={epoch} batch={idx} "
                            f"pgd_steps={used_pgd_steps} "
                            f"lang(mean/max)={lang_mean:.4f}/{lang_max:.4f} "
                            f"total(mean)={total_mean:.4f} "
                            f"l2={l2_val:.4f} "
                            f"delta_linf={delta_linf:.6f} r_linf={r_linf:.6f} "
                            f"snr={snr_db:.2f}dB"
                        )

            # --- 评估阶段：每隔 success_every 轮检查一次成功率 ---
            if (epoch % self.success_every) == 0:
                eval_result = self._summarize_language_shift(
                    loader, delta, desc=f"epoch={epoch}"
                )

                # 新增：记录 eval 历史
                self.eval_history.append({
                    "epoch": int(epoch),
                    "success_rate": float(eval_result["success_rate"]),
                    "mean_loss": float(eval_result["mean_loss"]),
                    "fooled": float(eval_result["fooled"]),
                    "total": float(eval_result["total"]),
                    "delta_linf": float(delta.abs().max().item()),
                    "lang_change_rate": float(eval_result["lang_change_rate"]),
                    "lang_changed": float(eval_result["lang_changed"]),
                    "lang_eval_total": float(eval_result["lang_eval_total"]),
                })

                # 保存最佳扰动
                if eval_result["success_rate"] > best_success_rate:
                    best_success_rate = eval_result["success_rate"]
                    self.univ_perturb.tensor.data = delta.detach()
                    self.checkpointer.save_and_keep_only()  # 保存 checkpoint

            # --- 语种预测观测：每隔 log_lang_pred_every 轮打印一次 ---
            if (
                self.log_lang_pred_every is not None
                and self.log_lang_pred_every > 0
                and (epoch % self.log_lang_pred_every) == 0
            ):
                self._log_language_predictions(loader, delta, epoch)

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
