from robust_speech.adversarial.attacks.pgd import SNRPGDAttack
from sb_whisper_binding import WhisperASR
from whisper_with_gradients import detect_language_with_gradients
from whisper.audio import CHUNK_LENGTH, SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram
import robust_speech as rs
import torch.nn as nn
import torch
from robust_speech.adversarial.utils import TargetGenerator
import copy 

def compute_forward_lang(whisper_asr_brain, batch, stage):
    assert stage == rs.Stage.ATTACK                           # 仅攻击阶段调用
    batch = batch.to(whisper_asr_brain.device)                # 将数据搬到设备
    wavs, wav_lens = batch.sig                                # 语音及长度
    tokens_bos, _ = batch.tokens_bos                          # BOS token（本函数不使用，但保持接口一致）

    if hasattr(whisper_asr_brain.hparams, "smoothing") and whisper_asr_brain.hparams.smoothing:
        wavs = whisper_asr_brain.hparams.smoothing(wavs, wav_lens)          # 随机平滑

    # 攻击阶段是否开启 env_corrupt 由 hparams.attack_use_env_corrupt 控制，默认关闭以节省显存
    use_env_corrupt = getattr(whisper_asr_brain.hparams, "attack_use_env_corrupt", False)
    if use_env_corrupt and hasattr(whisper_asr_brain.modules, "env_corrupt"):
        wavs_noise = whisper_asr_brain.modules.env_corrupt(wavs, wav_lens)   # 加噪增强
        wavs = torch.cat([wavs, wavs_noise], dim=0)                         # 拼接干净+噪声
        wav_lens = torch.cat([wav_lens, wav_lens])                          # 对应长度
        # 注意：tokens_bos 在本函数中不使用，因此不需要拼接以避免批次字段不一致

    # 同理，攻击阶段的其他增广也可通过 attack_use_augmentation 控制（默认关闭）
    use_aug = getattr(whisper_asr_brain.hparams, "attack_use_augmentation", False)
    if use_aug and hasattr(whisper_asr_brain.hparams, "augmentation"):
        wavs = whisper_asr_brain.hparams.augmentation(wavs, wav_lens)       # 其他增广

    # Forward pass
    # 注意：之前只取 batch 的第一条音频，导致批量评估时预测数量与样本数不一致，
    # 会把 success rate 分母做大、分子只统计单条预测。这里改为逐条音频计算语言预测，
    # 并在 batch 维度上堆叠返回，保证评估和成功率统计按真实样本数对齐。
    tokens, _ = batch.tokens                                               # 文本标签（未用，仅保持接口）

    use_autocast = getattr(whisper_asr_brain.hparams, "lang_autocast", True)

    # 逐条 mel 计算后一次性堆叠，避免在 Python for-loop 中重复构建/保留多份完整 encoder 计算图
    mels = []
    for audio in wavs:
        mel = log_mel_spectrogram(audio)                                  # 生成 mel
        mel = pad_or_trim(mel, N_FRAMES)                                  # 固定长度
        mels.append(mel)

    mel_batch = torch.stack(mels)                                         # (batch, n_mels, frames)
    language_tokens, _, logits = detect_language_with_gradients(
        whisper_asr_brain.modules.whisper.model, mel_batch, use_autocast=use_autocast
    )

    # detect_language_with_gradients 返回的 logits 需要保留梯度；概率用于日志则 detach+cpu 减少显存
    language_probs = torch.softmax(logits.detach(), dim=-1).cpu()
    language_tokens = language_tokens.to(whisper_asr_brain.device)
    logits = logits.to(whisper_asr_brain.device)

    return language_tokens, language_probs, logits

def compute_objectives_lang(
        whisper_asr_brain, lang_token, predictions, batch, stage, reduction="none",**kwargs
    ):
    assert stage == rs.Stage.ATTACK                                         # 仅攻击阶段
    language_tokens, language_probs, logits = predictions                   # 取预测
    target = lang_token.to(whisper_asr_brain.device)                        # 目标语言 token

    # CrossEntropy 要求目标 shape = (batch,)，此前只传单个 token，
    # 当 logits 含多条样本时会触发广播或仅对首样本生效，导致 loss/成功率统计偏差。
    if target.numel() == 1:
        targets = target.view(1).expand(logits.shape[0])                    # 每条样本同一目标语言
    else:
        # 极端情况：lang_token 本身是多 token，这里保守取第一 token；
        # 仍然保证长度与 batch 对齐，避免梯度只作用于单样本。
        targets = target.view(-1)[0].repeat(logits.shape[0])

    loss_fct = nn.CrossEntropyLoss(reduction=reduction)                     # 交叉熵
    loss = loss_fct(logits, targets)                                        # 语言分类损失
    return loss

class WhisperLangID(WhisperASR):
    def __init__(self,asr_brain,lang_token):
        assert isinstance(asr_brain,WhisperASR)
        self.asr_brain=asr_brain                                             # 保存原 brain
        self.device = self.asr_brain.device                                  # 设备
        self.modules = self.asr_brain.modules                                # 模型模块
        self.lang_token = lang_token                                         # 目标语言

    def compute_forward(self,*args,**kwargs):
        return compute_forward_lang(self.asr_brain,*args,**kwargs)
    def compute_objectives(self,*args,**kwargs):
        return compute_objectives_lang(self.asr_brain,self.lang_token,*args,**kwargs)

class WhisperLanguageAttack(SNRPGDAttack):
    def __init__(self,asr_brain,*args,language="es",targeted_for_language=True,**kwargs):
        self.language = "<|"+language.strip("<|>")+"|>"                       # 语言 token 文本
        self.lang_token = torch.LongTensor(asr_brain.tokenizer.encode(self.language))  # 转 token id
        super(WhisperLanguageAttack,self).__init__(WhisperLangID(asr_brain,self.lang_token),*args,targeted=targeted_for_language,**kwargs)
