# whisper_attack（Whisper 对抗攻击）

本仓库提供针对 Whisper ASR 模型的对抗样本生成与评估代码，对应论文：[There is more than one kind of robustness: Fooling Whisper with adversarial examples](https://arxiv.org/abs/2210.17316)。

我们提供复现论文实验的脚本，以及使用 huggingface transformers 在这些对抗样本上评估 Whisper 的示例。

## 依赖

- 安装 [robust_speech](https://github.com/RaphaelOlivier/robust_speech) 与 [whisper](https://github.com/openai/whisper)。
- 若使用 HF 推理流水线，需要 `transformers>=4.23.0`、`datasets>=2.5.0`、`evaluate>=0.2.2`。

## 用法

### 生成对抗样本
- `run_attack.py`：调用 robust_speech 的攻击评估脚本。  
  - `attack_configs/` 中的配置指定攻击类型、数据集与超参，可用命令行覆盖。  
  - `model_configs/` 中的配置指定各 Whisper 模型的加载信息。
- 论文中的 bash 复现脚本：
  - `cw.sh`：对 ASR 解码器的定向攻击。
  - `pgd.sh`：对 ASR 解码器的非定向攻击（35dB/40dB SNR）。
  - `smooth.sh`：在随机平滑防御下的非定向攻击。
  - `lang.sh`：针对语言检测器的定向攻击，劣化 ASR 表现（7 源语种、3 目标语种）。
  - `rand.sh`：加入高斯噪声作对比。

> 数据准备：  
> - 语言检测攻击使用 CommonVoice（相应源语言）。  
> - 其他攻击使用 LibriSpeech test-clean。  
> - 若只在子集上生成攻击，可先生成子集 csv，例如：  
>   ```head test-clean.csv -n 101 > test-clean-100.csv```

### 使用预计算的对抗样本
- `whisper_adversarial_examples/` 目录包含已计算好的 Huggingface 数据集，可直接配合 `inference.py` 使用，例如：
```
python inference.py --model whisper-medium.en --config untargeted-35
```
  更多示例见 `inference.sh`。
- 数据集也发布在 Huggingface Hub：<https://huggingface.co/datasets/RaphaelOlivier/whisper_adversarial_examples>  
  或直接下载压缩包：<https://data.mendeley.com/datasets/96dh52hz9r/draft?a=ee30841f-1832-41ec-bdac-bf3e5b67073c>。

## 许可证
**robust_speech** 以 Apache License 2.0 发布。

## 引用
若在实验中使用 robust_speech 或本仓库，请引用：

```bibtex
@article{Olivier2023FW,
  url = {https://arxiv.org/abs/2210.17316},
  author = {Olivier, Raphael and Raj, Bhiksha},
  title = {There is more than one kind of robustness: Fooling Whisper with adversarial examples},
  journal = {Interspeech},
  year = {2023},  
}
```
