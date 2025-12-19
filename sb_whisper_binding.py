import logging
import os
import sys

import speechbrain as sb
import torch
import torch.nn as nn
import string
from whisper_with_gradients import WhisperWithGradient

import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain

logger = logging.getLogger(__name__)


class WhisperASR(AdvASRBrain):
    """
    Whisper ASR model
    """

    def compute_forward(self, batch, stage):
        """前向：从波形到 logits/预测 token"""
        if not stage == rs.Stage.ATTACK:
            batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs = wavs.float().to(self.device)
        tokens_bos, _ = batch.tokens_bos
        # 推理配置与损失配置（如语言、fp16、beam）
        options = {
            "language": (
                self.hparams.language if hasattr(self.hparams, "language") else None
            ),
            "fp16": self.hparams.fp16 if hasattr(self.hparams, "fp16") else False,
            "without_timestamps": (
                self.hparams.without_timestamps
                if hasattr(self.hparams, "without_timestamps")
                else True
            ),
            "beam_size": (
                self.hparams.beam_size if hasattr(self.hparams, "beam_size") else None
            ),
        }
        loss_options = {
            "confidence": (
                self.hparams.confidence if hasattr(self.hparams, "confidence") else 0.0
            ),
            "correct_first_word": (
                self.hparams.correct_first_word
                if hasattr(self.hparams, "correct_first_word")
                else False
            ),
        }

        # if options["fp16"]:
        #     self.modules.to(torch.float16)
        # dtype = torch.float16 if options["fp16"] else torch.float32
        # ✅ 不要把模型整体 half（会把 LayerNorm 权重也 half，Whisper 会炸）
        self.modules.to(torch.float32)

        use_fp16 = bool(options["fp16"])
        dtype = torch.float16 if use_fp16 else torch.float32

        # 可选平滑/增广
        if hasattr(self.hparams, "smoothing") and self.hparams.smoothing:
            wavs = self.hparams.smoothing(wavs, wav_lens)
        if stage == sb.Stage.TRAIN or stage == rs.Stage.ATTACK:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        # 前向：训练/攻击直接用 loss；评估走 transcribe
        tokens, _ = batch.tokens
        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            with torch.no_grad():
                result = self.modules.whisper.model.loss(
                    wavs[0].to(dtype),
                    tokens[0],
                    task="transcribe",
                    **loss_options,
                    **options
                )
                loss = result["loss"].detach()
                result = self.modules.whisper.model.transcribe(
                    wavs[0], task="transcribe", **options
                )
                text = result["text"]
                pred_tokens = torch.LongTensor([self.tokenizer.encode(text)])
        else:
            result = self.modules.whisper.model.loss(
                wavs[0], tokens[0], task="transcribe", **loss_options, **options
            )
            loss = result["loss"]
            logits = result["logits"]
            pred_tokens = logits.argmax(dim=-1)
        return loss, pred_tokens, stage

    def get_tokens(self, predictions):
        """根据阶段选择输出 token"""
        if predictions[2] in [sb.Stage.VALID, sb.Stage.TEST]:
            tokens = predictions[1].cpu()
        else:
            tokens = predictions[1][:, :-1].cpu()
        return tokens

    def compute_objectives(
        self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"
    ):
        """计算 WER/CER 指标或返回损失（训练/攻击阶段直接用 loss）"""

        loss, pred_tokens, save_stage = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat([tokens_eos_lens, tokens_eos_lens], dim=0)
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            # 解码成单词并去除标点以计算 WER/CER
            predicted_words = [
                self.tokenizer.decode(t)
                .strip()
                .upper()
                .translate(str.maketrans("", "", string.punctuation))
                for t in pred_tokens
            ]
            predicted_words = [wrd.split(" ") for wrd in predicted_words]
            target_words = [
                wrd.upper()
                .translate(str.maketrans("", "", string.punctuation))
                .split(" ")
                for wrd in batch.wrd
            ]

            if adv:
                if targeted:
                    self.adv_wer_metric_target.append(
                        ids, predicted_words, target_words
                    )
                    self.adv_cer_metric_target.append(
                        ids, predicted_words, target_words
                    )
                    self.adv_ser_metric_target.append(
                        ids, predicted_words, target_words
                    )
                else:
                    self.adv_wer_metric.append(ids, predicted_words, target_words)
                    self.adv_cer_metric.append(ids, predicted_words, target_words)
            else:
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)
        return loss

    def init_optimizers(self):
        "Initializes the optimizer and model optimizer"
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)
