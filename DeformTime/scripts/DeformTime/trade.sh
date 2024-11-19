#!/bin/bash -l
#SBATCH -J deform_sp500
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/sp500a/test_MS_1.o
#SBATCH -e ./logs/sp500a/test_MS_1.e
#SBATCH -A cps -p tier3 -n 4
#SBATCH --mem=8GB
#SBATCH --gres=gpu:a100:1

#spack load cuda@10.2.89 /ydlu6td # cuda 11.8
#spack load gcc@8.2.0 /r54eocp # gcc 12.2

spack load /igzaycn # cuda 11.8
spack load /dr4ipev # gcc 12.2

export CUDA_VISIBLE_DEVICES=0

model_name=DeformTime

source ../../miniconda/etc/profile.d/conda.sh
conda activate ptorch


python3 -u run_trades.py \
  --is_training 1 \
  --root_path ./data/dataset/SP500/ \
  --ticker_file tickers.txt \
  --model_id SP500a_1 \
  --model $model_name \
  --data SP500a \
  --data_path stocks/all.csv \
  --features MS \
  --target RET \
  --inverse \
  --seq_len 336 \
  --pred_len 1 \
  --label_len 0 \
  --batch_size 16 \
  --d_model 32 \
  --learning_rate 0.001 \
  --train_epochs 1 \
  --dropout 0 \
  --layer_dropout 0 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 300 \
  --dec_in 300 \
  --c_out 300 \
  --n_heads 4 \
  --n_reshape 24 \
  --patch_len 6 \
  --kernel 7 \
  --patience 10 \
  --des 'Exp' \
  --loss Stock \
  --itr 1

# python3 -u run.py \
#   --is_training 1 \
#   --root_path ./data/dataset/ETT/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_6 \
#   --model $model_name \
#   --data ETTh1 \
#   --features MS \
#   --target OT \
#   --seq_len 336 \
#   --pred_len 6 \
#   --label_len 0 \
#   --batch_size 16 \
#   --d_model 32 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --dropout 0 \
#   --layer_dropout 0 \
#   --e_layers 2 \
#   --d_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --n_heads 4 \
#   --n_reshape 6 \
#   --patch_len 24 \
#   --kernel 3 \
#   --patience 10 \
#   --des 'Exp' \
#   --itr 10

# python3 -u run.py \
#   --is_training 0 \
#   --root_path ./data/dataset/ETT/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_12 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 336 \
#   --pred_len 12 \
#   --label_len 0 \
#   --batch_size 16 \
#   --d_model 32 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --dropout 0 \
#   --layer_dropout 0 \
#   --e_layers 2 \
#   --d_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --n_heads 4 \
#   --n_reshape 6 \
#   --patch_len 24 \
#   --kernel 3 \
#   --patience 10 \
#   --des 'Exp' \
#   --itr 10

# python3 -u run.py \
#   --is_training 0 \
#   --root_path ./data/dataset/ETT/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_24 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 336 \
#   --pred_len 24 \
#   --label_len 0 \
#   --batch_size 16 \
#   --d_model 32 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --dropout 0 \
#   --layer_dropout 0 \
#   --e_layers 2 \
#   --d_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --n_heads 4 \
#   --n_reshape 6 \
#   --patch_len 24 \
#   --kernel 3 \
#   --patience 10 \
#   --des 'Exp' \
#   --itr 10