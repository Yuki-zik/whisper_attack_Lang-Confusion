#!/bin/bash
# 语言检测定向攻击与通用语言攻击示例，保持原参数与命令

source activate sp3
NBITER=30
SEED=1030
LOAD=True
NAME=lang
DATA=test-100
MODEL=medium
SNR=45

# 下方多语言攻击示例已注释（保留原命令方便启用）：
# LANGATTACK=en
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=hy-AM --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# ...
# LANGATTACK=tl / sr 等配置同上

MODEL=medium
LANGATTACK=sr
EPOCHS=10000
NBITER=1
SEED=1101
RELEPS=0.01
EVERY=100
DATA="test-90 --output_folder=/home/rolivier/workhorse1/robust_speech/attacks/univ_lang/sr/whisper-small/1101"
# 通用语言攻击训练示例（保持注释）
# python fit_attacker.py attack_configs/whisper/univ_lang_fit.yaml --lang_CV=it --lang_attack=$LANGATTACK --epochs=$EPOCHS --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=0.005 --batch_size=1 --rel_eps_iter=$RELEPS --success_every=$EVERY

# 通用语言攻击评估（不同源语言）
python run_attack.py attack_configs/whisper/univ_lang.yaml --lang_CV=it --lang_attack=$LANGATTACK --model_label=$MODEL --data_csv_name=$DATA --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=0.005 --params_subfolder=CKPT+2023-03-05+06-00-05+00
python run_attack.py attack_configs/whisper/univ_lang.yaml --lang_CV=en --lang_attack=$LANGATTACK --model_label=$MODEL --data_csv_name=$DATA --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=0.005 --params_subfolder=CKPT+2023-03-05+06-00-05+00
python run_attack.py attack_configs/whisper/univ_lang.yaml --lang_CV=da --lang_attack=$LANGATTACK --model_label=$MODEL --data_csv_name=$DATA --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=0.005 --params_subfolder=CKPT+2023-03-05+06-00-05+00
python run_attack.py attack_configs/whisper/univ_lang.yaml --lang_CV=hy-AM --lang_attack=$LANGATTACK --model_label=$MODEL --data_csv_name=$DATA --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=0.005 --params_subfolder=CKPT+2023-03-05+06-00-05+00
python run_attack.py attack_configs/whisper/univ_lang.yaml --lang_CV=id --lang_attack=$LANGATTACK --model_label=$MODEL --data_csv_name=$DATA --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=0.005 --params_subfolder=CKPT+2023-03-05+06-00-05+00
python run_attack.py attack_configs/whisper/univ_lang.yaml --lang_CV=cs --lang_attack=$LANGATTACK --model_label=$MODEL --data_csv_name=$DATA --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=0.005 --params_subfolder=CKPT+2023-03-05+06-00-05+00
python run_attack.py attack_configs/whisper/univ_lang.yaml --lang_CV=lt --lang_attack=$LANGATTACK --model_label=$MODEL --data_csv_name=$DATA --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=0.005 --params_subfolder=CKPT+2023-03-05+06-00-05+00
