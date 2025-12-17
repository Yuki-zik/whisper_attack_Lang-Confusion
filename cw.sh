#!/bin/bash
# Carlini&Wagner 定向攻击，多模型批量运行示例（参数与原脚本一致）

NBITER=2000         # 迭代次数
SEED=2000           # 随机种子
LOAD=False          # 是否加载音频
NAME=cw             # 攻击名称
EPS=0.1             # Linf 边界
MAXDECR=8           # eps 衰减次数
DATA=test-clean-20  # 数据子集 CSV 名
CONF=0.0            # 置信度参数
DECRFACTOR=0.7      # eps 衰减因子
CST=4               # C&W 常数
LR=0.01             # 学习率

python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=large --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
