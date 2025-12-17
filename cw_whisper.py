# Carlini&Wagner attack (https://arxiv.org/abs/1801.01944)
# 适配语音 ASR，基于 robust_speech 的 ImperceptibleASRAttack

from typing import List, Optional, Tuple

import numpy as np
import speechbrain as sb
import torch
import torch.nn as nn
import torch.optim as optim

import robust_speech as rs
from robust_speech.adversarial.attacks.imperceptible import ImperceptibleASRAttack


class ASRCarliniWagnerAttack(ImperceptibleASRAttack):
    """
    A Carlini&Wagner attack for ASR models.
    The algorithm follows the first attack in https://arxiv.org/abs/1801.01944
    Based on the ART implementation of Imperceptible
    (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/imperceptible_asr/imperceptible_asr_pytorch.py)

    Arguments
    ---------
     asr_brain : rs.adversarial.brain.ASRBrain
        the brain object to attack
     targeted: bool
        if the attack is targeted (always true for now).
     eps: float
        Linf bound applied to the perturbation.
     learning_rate: float
        the learning rate for the attack algorithm
     max_iter: int
        the maximum number of iterations
     clip_min: float
        mininum value per input dimension (ignored: herefor compatibility).
     clip_max: float
        maximum value per input dimension (ignored: herefor compatibility).
     train_mode_for_backward: bool
        whether to force training mode in backward passes (necessary for RNN models)
    global_max_length: int
        max length of a perturbation
    initial_rescale: float
        initial factor by which to rescale the perturbation
    num_iter_decrease_eps: int
        Number of times to increase epsilon in case of success
    decrease_factor_eps: int
        Factor by which to decrease epsilon in case of failure
    optimizer: Optional["torch.optim.Optimizer"]
        the optimizer to use
    """

    def __init__(
        self,
        asr_brain: rs.adversarial.brain.ASRBrain,
        eps: float = 0.05,
        max_iter: int = 10,
        learning_rate: float = 0.001,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        global_max_length: int = 200000,
        initial_rescale: float = 1.0,
        decrease_factor_eps: float = 0.8,
        num_iter_decrease_eps: int = 1,
        max_num_decrease_eps: Optional[int] = None,
        targeted: bool = True,
        train_mode_for_backward: bool = True,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        const: float = 1.0,
        confidence: float = 0.0,
        correct_first_word: bool = False
    ):
        # 调用父类构造：第一阶段迭代 max_iter，第二阶段禁用（max_iter_2=0）
        super(ASRCarliniWagnerAttack, self).__init__(
            asr_brain,
            eps=eps,
            max_iter_1=max_iter,
            max_iter_2=0,
            learning_rate_1=learning_rate,
            optimizer_1=optimizer,
            global_max_length=global_max_length,
            initial_rescale=initial_rescale,
            decrease_factor_eps=decrease_factor_eps,
            num_iter_decrease_eps=num_iter_decrease_eps,
            max_num_decrease_eps=max_num_decrease_eps,
            targeted=targeted,
            train_mode_for_backward=train_mode_for_backward,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.reg_const = 1./const if const is not None else 0.  # C&W 正则系数
        self.confidence = confidence                              # 调整 logits 的置信度阈
        self.correct_first_word = correct_first_word              # 是否修正首词
        self.asr_brain.hparams.confidence = self.confidence       # 将参数写入 brain
        self.asr_brain.hparams.correct_first_word = self.correct_first_word

    def _forward_1st_stage(
        self,
        original_input: np.ndarray,
        batch: sb.dataio.batch.PaddedBatch,
        local_batch_size: int,
        local_max_length: int,
        rescale: np.ndarray,
        input_mask: np.ndarray,
        real_lengths: np.ndarray,
    ):

        # 取当前 batch 的扰动并裁剪到 [-eps, eps]
        local_delta = self.global_optimal_delta[:
                                                local_batch_size, :local_max_length]
        local_delta_rescale = torch.clamp(local_delta, -self.eps, self.eps).to(
            self.asr_brain.device
        )
        local_delta_rescale *= torch.tensor(rescale).to(self.asr_brain.device)
        # 叠加到原始输入得到对抗音频
        adv_input = local_delta_rescale + torch.tensor(original_input).to(
            self.asr_brain.device
        )
        # mask 去除填充位置
        masked_adv_input = adv_input * torch.tensor(input_mask).to(
            self.asr_brain.device
        )

        # 用对抗音频替换 batch 后前向
        batch.sig = masked_adv_input, batch.sig[1]
        predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)
        loss = self.asr_brain.compute_objectives(
            predictions, batch, rs.Stage.ATTACK, reduction="batchmean")
        # 增加扰动范数正则
        loss_backward = loss.mean() + self.reg_const * torch.norm(local_delta_rescale)
        decoded_output = self.asr_brain.get_tokens(predictions)
        # print(decoded_output,batch.tokens)
        # if teacher forcing prediction is correct, check decoder transcription
        if (decoded_output[0].view(-1) == batch.tokens_eos[0].cpu().view(-1)).all():
            self.asr_brain.module_eval()
            val_predictions = self.asr_brain.compute_forward(
                batch, sb.Stage.VALID)
            val_decoded_output = self.asr_brain.get_tokens(val_predictions)
            decoded_output = val_decoded_output
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        if len(loss.size()) == 0:
            loss = loss.view(1)
        # 返回攻击损失、当前扰动等
        return loss_backward, loss, local_delta, decoded_output, masked_adv_input, local_delta_rescale
