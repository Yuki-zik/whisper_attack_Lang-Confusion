"""
对抗攻击评估脚本：与训练类似但不调用 brain.fit()。
为支持迁移攻击 / 多模型（如 MGAA），模型 hparams 与主配置解耦。

hparams 中的 target_brain_class/target_brain_hparams_file 用于加载目标 brain；
可选 source_brain_class/source_brain_hparams_file 用于迁移扰动，支持嵌套列表构造 EnsembleASRBrain。
示例：
python run_attack.py attack_configs/pgd/attack.yaml --root=/path/to/data --auto_mix_prec
"""
import os
import sys
from pathlib import Path

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain
from robust_speech.adversarial.utils import TargetGeneratorFromFixedTargets


def read_brains(
    brain_classes,
    brain_hparams,
    attacker=None,
    run_opts={},
    overrides={},
    tokenizer=None,
):
    # 支持单 brain 或列表（列表时递归并封装 EnsembleASRBrain）
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
        if isinstance(brain_hparams, str):  # 如为路径则先读取 yaml
            with open(brain_hparams) as fin:
                brain_hparams = load_hyperpyyaml(fin, overrides)
        checkpointer = (
            brain_hparams["checkpointer"] if "checkpointer" in brain_hparams else None
        )
        # 实例化单个 brain，可传 attacker
        brain = brain_classes(
            modules=brain_hparams["modules"],
            hparams=brain_hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
            attacker=attacker,
        )
        if "pretrainer" in brain_hparams:  # 预训练权重加载
            run_on_main(brain_hparams["pretrainer"].collect_files)
            brain_hparams["pretrainer"].load_collected(
                device=run_opts["device"])
        brain.tokenizer = tokenizer  # 共享 tokenizer
    return brain


def evaluate(hparams_file, run_opts, overrides):
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
        # tokenizer 由主 hparams 提供，加载预训练权重
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
    if "tokenizer_builder" in hparams:
        tokenizer = hparams["tokenizer_builder"](hparams["tokenizer_name"])
        hparams["tokenizer"] = tokenizer
    else:
        tokenizer=hparams["tokenizer"]

    # 构建数据集与编码
    _, _, test_datasets, _, _, tokenizer = dataio_prepare(hparams)
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
    # Evaluation
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

    evaluate(hparams_file, run_opts, overrides)
