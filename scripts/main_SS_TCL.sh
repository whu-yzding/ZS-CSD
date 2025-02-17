#!/bin/bash
currTime=$(date +"%Y-%m-%d_%T")
seed=42
nohup python main.py --seed $seed --alpha 1.0 --weight_decay 1e-6 --tau 0.05  --cuda_index 1 --gru_layer 1 --gru_hidden 768 --time $currTime  > "./tmp/Ours_SS_TCL/$currTime.log" 2>&1 &