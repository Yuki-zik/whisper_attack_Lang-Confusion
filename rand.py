import torch
import numpy as np
from robust_speech.adversarial.defenses.smoothing import SpeechNoiseAugmentation  # 加噪防御/增广
from robust_speech.adversarial.attacks.attacker import Attacker                   # 攻击基类


class GaussianAttack(Attacker):
    """简单高斯噪声攻击，用于对比实验"""
    def __init__(self, asr_brain, sigma=0, **kwargs):
        self.asr_brain = asr_brain
        self.smoother = SpeechNoiseAugmentation(sigma=sigma)  # 生成高斯噪声

    def perturb(self, batch):
        wav = self.smoother.forward(batch.sig[0], batch.sig[1])  # 在波形上加噪
        return wav
