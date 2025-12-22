"""
对抗扰动训练脚本：与评估类似，但调用 fit_attacker 训练攻击器。
支持迁移/多模型设置（模型 hparams 与主配置解耦）。
"""
import csv
import os
import sys
from pathlib import Path

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.depgraph import CircularDependencyError
import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain
from robust_speech.adversarial.utils import TargetGeneratorFromFixedTargets
import logging

logger = logging.getLogger("speechbrain.utils.epoch_loop")
logger.setLevel(logging.WARNING)

def read_brains(
    brain_classes,
    brain_hparams,
    attacker=None,
    run_opts={},
    overrides={},
    tokenizer=None,
):
    # 支持列表（构建 EnsembleASRBrain）或单个 brain
    if isinstance(brain_classes, list):
        brain_list = []
        assert len(brain_classes) == len(brain_hparams)
        for bc, bf in zip(brain_classes, brain_hparams):
            br = read_brains(
                bc, bf, run_opts=run_opts, overrides=overrides, tokenizer=tokenizer
            )
            brain_list.append(br)
        brain = rs.adversarial.brain.EnsembleASRBrain(brain_list)
    else:
        if isinstance(brain_hparams, str):  # yaml 路径则先读取
            with open(brain_hparams) as fin:
                brain_hparams = load_hyperpyyaml(fin, overrides)
        checkpointer = (
            brain_hparams["checkpointer"] if "checkpointer" in brain_hparams else None
        )
        # 实例化 brain，可传入攻击器
        brain = brain_classes(
            modules=brain_hparams["modules"],
            hparams=brain_hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
            attacker=attacker,
        )
        if "pretrainer" in brain_hparams:  # 预训练权重
            run_on_main(brain_hparams["pretrainer"].collect_files)
            brain_hparams["pretrainer"].load_collected(
                device=run_opts["device"])
        brain.tokenizer = tokenizer  # 共享 tokenizer
    return brain


def fit(hparams_file, run_opts, overrides):
    # 读取主 hparams
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if "pretrainer" in hparams:  # load parameters
        # tokenizer 从主 hparams 共享给各 brain
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Dataset prep (parsing Librispeech)
    prepare_dataset = hparams["dataset_prepare_fct"]  # 数据准备函数
    if (
        hparams.get("test_splits") == ["test"]
        and "test_csv" in hparams
        and "librispeech" in str(hparams["dataset_prepare_fct"]).lower()
    ):
        test_csv = hparams["test_csv"]
        if isinstance(test_csv, str):
            test_csv = [test_csv]
        hparams["test_splits"] = [Path(csv_path).stem for csv_path in test_csv]
    if isinstance(prepare_dataset, str):
        # allow passing import path via CLI overrides (loses !name tag)
        module_path, fn_name = prepare_dataset.rsplit(".", 1)
        prepare_dataset = getattr(__import__(module_path, fromlist=[fn_name]), fn_name)
    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_dataset,
        kwargs={
            "data_folder": hparams["data_folder"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["csv_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    dataio_prepare = hparams["dataio_prepare_fct"]
    # 确保测试相关的配置在 CLI 覆盖时仍保持列表格式。
    # SpeechBrain 的 dataio_prepare 会用 zip(test_splits, test_csv) 逐项打包，
    # 每个 split 与一条 CSV 一一对应：例如 test_splits=["test_clean", "test_other"]
    # 且 test_csv=[clean.csv, other.csv] 时会得到两个独立的测试集，稍后评估时
    # 会分别调用 evaluate，而不是把所有测试样本合并后一次性跑完。若任一参数
    # 以字符串形式传入，zip 会把字符串拆成字符迭代并尝试打开 "/" 等非法路径，
    # 因此在这里把字符串归一化为列表，避免再次触发 IsADirectoryError。
    if isinstance(hparams.get("test_csv"), str):
        hparams["test_csv"] = [hparams["test_csv"]]
    if isinstance(hparams.get("test_splits"), str):
        hparams["test_splits"] = [hparams["test_splits"]]
    if isinstance(hparams.get("test_csv"), str):
        hparams["test_csv"] = [hparams["test_csv"]]
    if "tokenizer_builder" in hparams:
        tokenizer = hparams["tokenizer_builder"](hparams["tokenizer_name"])
        hparams["tokenizer"] = tokenizer
    else:
        tokenizer=hparams["tokenizer"]

    # 在构建数据集前提前验证 CSV 文件，避免因缺失必备列而触发循环依赖。
    def _collect_columns(csv_path):
        path_obj = Path(csv_path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"CSV 文件不存在或不可读：{csv_path}")
        with path_obj.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        return set(header or [])

    def _validate_csv_columns(csv_list):
        expected = {"ID", "duration", "wav", "wrd"}
        if hparams.get("targeted_for_language", False):
            expected.add("lang")
        for csv_path in csv_list:
            columns = _collect_columns(csv_path)
            missing = expected - columns
            if missing:
                raise ValueError(
                    f"CSV {csv_path} 缺少必要列：{sorted(missing)}，"
                    "请补充 wav 路径与转写文本（语言攻击还需 lang 列）。"
                )

    train_csv = hparams.get("train_csv")
    if train_csv:
        _validate_csv_columns([train_csv])
    _validate_csv_columns(hparams.get("test_csv", []))

    # 构建数据集对象与编码
    train_dataset, _, test_datasets, _, _, tokenizer = dataio_prepare(hparams)
    source_brain = None
    if "source_brain_class" in hparams:  # loading source model
        source_brain = read_brains(
            hparams["source_brain_class"],
            hparams["source_brain_hparams_file"],
            run_opts=run_opts,
            overrides={"root": hparams["root"]},
            tokenizer=tokenizer,
        )
    attacker = hparams["attack_class"]  # 攻击器类型
    if source_brain and attacker:
        # instanciating with the source model if there is one.
        # Otherwise, AdvASRBrain will handle instanciating the attacker with
        # the target model.
        if isinstance(source_brain, rs.adversarial.brain.EnsembleASRBrain):
            if "source_ref_attack" in hparams:
                source_brain.ref_attack = hparams["source_ref_attack"]
            if "source_ref_train" in hparams:
                source_brain.ref_train = hparams["source_ref_train"]
            if "source_ref_valid_test" in hparams:
                source_brain.ref_valid_test = hparams["source_ref_valid_test"]
        attacker = attacker(source_brain)

    # Target model initialization
    target_brain_class = hparams["target_brain_class"]
    target_hparams = (
        hparams["target_brain_hparams_file"]
        if hparams["target_brain_hparams_file"]
        else hparams
    )
    if source_brain is not None and target_hparams == hparams["source_brain_hparams_file"]:
        # avoid loading the same network twice
        sc_brain = source_brain
        sc_class = target_brain_class
        if isinstance(source_brain, rs.adversarial.brain.EnsembleASRBrain):
            sc_brain = source_brain.asr_brains[source_brain.ref_valid_test]
            sc_class = target_brain_class[source_brain.ref_valid_test]
        target_brain = sc_class(
            modules=sc_brain.modules,
            hparams=sc_brain.hparams.__dict__,
            run_opts=run_opts,
            checkpointer=None,
            attacker=attacker,
        )
        target_brain.tokenizer = tokenizer
    else:
        target_brain = read_brains(
            target_brain_class,
            target_hparams,
            attacker=attacker,
            run_opts=run_opts,
            overrides={"root": hparams["root"]},
            tokenizer=tokenizer,
        )
    target_brain.logger = hparams["logger"]
    target_brain.hparams.train_logger = hparams["logger"]
    #attacker.other_asr_brain = target_brain
    target = None
    if "target_generator" in hparams:
        target = hparams["target_generator"]
    elif "target_sentence" in hparams:
        target = TargetGeneratorFromFixedTargets(
            target=hparams["target_sentence"])
    load_audio = hparams["load_audio"] if "load_audio" in hparams else None
    save_audio_path = hparams["save_audio_path"] if hparams["save_audio"] else None
    # Training：恢复 checkpoint（如存在），再训练攻击器
    checkpointer = hparams["checkpointer"]
    target_brain.attacker.checkpointer=checkpointer
    checkpointer.recover_if_possible(
                device=run_opts["device"]
            )
    try:
        target_brain.fit_attacker(
            train_dataset,
            loader_kwargs=hparams["train_dataloader_opts"],
        )
    except CircularDependencyError as exc:  # pragma: no cover - runtime safeguard
        # 当数据管线的动态项存在环依赖（常见于 CSV 列缺失或重复自定义管线）时，SpeechBrain 会抛出该异常。
        # 这里读取 train/test CSV 的表头，帮助定位是哪些字段缺失或命名不符。
        csv_paths = []
        train_csv = hparams.get("train_csv")
        if train_csv:
            train_csv = [train_csv] if isinstance(train_csv, str) else list(train_csv)
            csv_paths.extend(train_csv)
        for item in hparams.get("test_csv", []):
            csv_paths.append(item)
        columns = set()
        missing_files = []
        for csv_path in csv_paths:
            path_obj = Path(csv_path)
            if not path_obj.is_file():
                missing_files.append(str(csv_path))
                continue
            with path_obj.open(newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    columns.update(header)
        msg_parts = [
            "检测到数据管线的循环依赖，通常由 CSV 缺少必要列（wav/wrd/lang）或自定义管线重复命名导致。",
            f"当前已识别的 CSV 表头：{sorted(columns) if columns else '（未读到表头）'}。",
        ]
        if missing_files:
            msg_parts.append(f"以下 CSV 路径不存在或不可读：{missing_files}。")
        msg_parts.append("请检查 train_csv/test_csv 内容与 dataio_prepare 配置是否匹配（例如是否包含 wav 路径、文本/语言标签列）。")
        raise RuntimeError("".join(msg_parts)) from exc

    # saving parameters
    checkpointer.save_checkpoint()
    # Evaluation：训练后在测试集上评估
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        target_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        target_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_dataloader_opts"],
            load_audio=load_audio,
            save_audio_path=save_audio_path,
            sample_rate=hparams["sample_rate"],
            target=target,
        )


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    fit(hparams_file, run_opts, overrides)
