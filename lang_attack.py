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
    tokens_bos, _ = batch.tokens_bos                          # BOS token

    if hasattr(whisper_asr_brain.hparams, "smoothing") and whisper_asr_brain.hparams.smoothing:
        wavs = whisper_asr_brain.hparams.smoothing(wavs, wav_lens)          # 随机平滑

    if hasattr(whisper_asr_brain.modules, "env_corrupt"):
        wavs_noise = whisper_asr_brain.modules.env_corrupt(wavs, wav_lens)   # 加噪增强
        wavs = torch.cat([wavs, wavs_noise], dim=0)                         # 拼接干净+噪声
        wav_lens = torch.cat([wav_lens, wav_lens])                          # 对应长度
        tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)             # 对齐 BOS

    if hasattr(whisper_asr_brain.hparams, "augmentation"):
        wavs = whisper_asr_brain.hparams.augmentation(wavs, wav_lens)       # 其他增广
    # Forward pass
    tokens, _ = batch.tokens                                               # 文本标签
    audio = wavs[0]                                                        # 取第一条音频
    mel = log_mel_spectrogram(audio)                                      # 生成 mel
    mel = pad_or_trim(mel,N_FRAMES)                                       # 固定长度
    language_tokens, language_probs, logits = detect_language_with_gradients(
        whisper_asr_brain.modules.whisper.model,mel
    )
    return language_tokens, language_probs, logits

def compute_objectives_lang(
        whisper_asr_brain, lang_token, predictions, batch, stage, reduction="none",**kwargs
    ):
    assert stage == rs.Stage.ATTACK                                         # 仅攻击阶段
    language_tokens, language_probs, logits = predictions                   # 取预测
    tokens=lang_token.to(whisper_asr_brain.device)                          # 目标语言 token
    #print(language_tokens, logits[0,language_tokens.item()],logits[0,tokens.item()])
    loss_fct = nn.CrossEntropyLoss(reduction=reduction)                     # 交叉熵
    loss = loss_fct(logits,tokens)                                          # 语言分类损失
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
