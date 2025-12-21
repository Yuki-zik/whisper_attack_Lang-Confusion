# whisper_attack（Whisper 对抗攻击）

本仓库提供针对 Whisper ASR 模型的对抗样本生成与评估代码，对应论文：[There is more than one kind of robustness: Fooling Whisper with adversarial examples](https://arxiv.org/abs/2210.17316)。

我们提供复现论文实验的脚本，以及使用 huggingface transformers 在这些对抗样本上评估 Whisper 的示例。

## 依赖

- 安装 [robust_speech](https://github.com/RaphaelOlivier/robust_speech) 与 [whisper](https://github.com/openai/whisper)。
- 若使用 HF 推理流水线，需要 `transformers>=4.23.0`、`datasets>=2.5.0`、`evaluate>=0.2.2`。

### 安装步骤
- 安装 **robust_speech**（最新版）
  ```bash
  pip install "git+https://github.com/RaphaelOlivier/robust_speech.git"
  ```
  若需固定版本，可手动克隆并安装：
  ```bash
  git clone https://github.com/RaphaelOlivier/robust_speech.git
  cd robust_speech
  git checkout <commit_sha>
  pip install .
  ```

- 安装 **OpenAI Whisper**（PyPI 包）
  ```bash
  pip install -U openai-whisper
  ```
  如需主分支版本，可使用：
  ```bash
  pip install "git+https://github.com/openai/whisper.git"
  ```

> 提示：Whisper 依赖 `ffmpeg` 用于音频解码，可通过 `sudo apt-get install ffmpeg` 安装。若 `git` 下载受到网络限制，可先手动下载源码压缩包再执行 `pip install .`。

### 安装故障排查
- 若想“只改少量设置就把安装跑通”，可以按以下顺序操作，成功后即可停止：
  1. **只影响当前命令的方式**：
     ```bash
     GIT_HTTP_VERSION=HTTP/1.1 pip install "git+https://github.com/RaphaelOlivier/robust_speech.git"
     ```
  2. **需要持续生效时**：
     ```bash
     git config --global http.version HTTP/1.1
     pip install "git+https://github.com/RaphaelOlivier/robust_speech.git"
     ```
  3. **网络仍不稳定时**：在能联网的环境下载源码压缩包（或让他人帮忙下载后拷贝），解压并在目录内执行：
     ```bash
     pip install .
     ```
- 若在执行 `pip install "git+https://github.com/RaphaelOlivier/robust_speech.git"` 时出现类似
  ```
  error: RPC failed; curl 16 Error in the HTTP2 framing layer
  fatal: expected flush after ref listing
  ```
  这通常是网络或代理对 HTTP/2 支持不稳定导致的 `git clone` 失败。可尝试：
  - 强制使用 HTTP/1.1 再安装：
    ```bash
    GIT_HTTP_VERSION=HTTP/1.1 pip install "git+https://github.com/RaphaelOlivier/robust_speech.git"
    ```
  - 或持久修改 Git 配置后重试：
    ```bash
    git config --global http.version HTTP/1.1
    pip install "git+https://github.com/RaphaelOlivier/robust_speech.git"
    ```
  - 若网络仍受限，手动下载源码压缩包（或在可访问环境下下载后传入本机），解压并在目录内执行 `pip install .`。

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

#### 通用语种攻击（Universal language attack）
- 训练通用扰动（例：源语种=意大利语，目标语种=塞尔维亚语，Whisper-medium）：
  ```bash
  # 需在 robust_speech 的数据根目录下指定 CommonVoice 子集路径
  python fit_attacker.py attack_configs/whisper/univ_lang_fit.yaml \
    --lang_CV=it --lang_attack=sr --model_label=medium \
    --data_csv_name="test-90 --output_folder=/path/to/robust_speech/attacks/univ_lang/sr/whisper-medium/1101" \
    --root=$RSROOT --load_audio=True --seed=1101 --nb_iter=1 --eps=0.005 \
    --epochs=10000 --batch_size=1 --rel_eps_iter=0.01 --success_every=100
  ```
- 评估通用扰动（使用上一步生成的扰动，切换不同源语种）：
  ```bash
  python run_attack.py attack_configs/whisper/univ_lang.yaml \
    --lang_CV=en --lang_attack=sr --model_label=medium --data_csv_name=test-90 \
    --root=$RSROOT --load_audio=True --seed=1101 --nb_iter=1 --eps=0.005 \
    --params_subfolder=CKPT+2023-03-05+06-00-05+00
  ```
- 如需批量评估多语种，可参考 `lang.sh` 末尾的多行 `run_attack.py` 示例，修改 `--lang_CV` 以切换源语种；`--lang_attack` 控制目标语言。

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



---
# Whisper 对抗攻击（Whisper Attack）uap

 python fit_attacker.py attack_configs/whisper/univ_lang_fit.yaml \
    --root=/root/autodl-tmp/whisper_attack_Lang-Confusion \
    --dataset_prepare_fct=robust_speech.data.librispeech.prepare_librispeech \
    --data_folder=/root/autodl-tmp/prepend_acoustic_attack/data/librispeech/LibriSpeech \
    --csv_folder=/root/autodl-tmp/prepend_acoustic_attack/data/librispeech/LibriSpeech/csv \
    --train_csv=/root/autodl-tmp/prepend_acoustic_attack/data/librispeech/LibriSpeech/csv/fit.csv \
    --test_csv=/root/autodl-tmp/prepend_acoustic_attack/data/librispeech/LibriSpeech/csv/test-clean.csv \
    --lang_attack=fr --lang_CV=en \
    --model_label=small \
    --load_audio=False \
    --batch_size=32 --nb_iter=10 --eps=0.02 --eps_item=0.01 \
    --rel_eps_iter=0.1 --epochs=5000 --success_every=100 --seed=1101

  python fit_attacker.py attack_configs/whisper/univ_lang_fit.yaml \
    --lang_CV=it --lang_attack=sr --model_label=medium \
    --data_csv_name="test-90 --output_folder=/path/to/robust_speech/attacks/univ_lang/sr/whisper-medium/1101" \
    --root=$RSROOT --load_audio=True --seed=1101 --nb_iter=1 --eps=0.005 \
    --epochs=10000 --batch_size=1 --rel_eps_iter=0.01 --success_every=100


```bash
# 生成训练用 CSV
python csv_make.py \
  --split-path .../dev-clean \
  --role fit \
  --lang en --compute-duration

# 生成测试用 CSVda
python csv_make.py \
  --split-path /root/autodl-tmp/prepend_acoustic_attack/data/librispeech/LibriSpeech/test-clean \
  --role testclean \
  --lang en --compute-duration
```

---
