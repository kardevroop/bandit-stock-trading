#!/bin/bash -l
#SBATCH -J de_illness
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/ILI/test_M.o
#SBATCH -e ./logs/ILI/test_M.e
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


python3 -u run.py \
  --is_training 0 \
  --root_path ./data/dataset/ILI/ \
  --data_path national_illness.csv \
  --model_id ILI_1 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 28 \
  --pred_len 1 \
  --label_len 0 \
  --batch_size 64 \
  --d_model 64 \
  --learning_rate 0.0001 \
  --train_epochs 50 \
  --dropout 0 \
  --layer_dropout 0 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 4 \
  --n_reshape 12 \
  --patch_len 24 \
  --kernel 9 \
  --patience 10 \
  --des 'Exp' \
  --itr 10



python3 -u run.py \
  --is_training 0 \
  --root_path ./data/dataset/ILI/ \
  --data_path national_illness.csv \
  --model_id ILI_6 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 28 \
  --pred_len 6 \
  --label_len 0 \
  --batch_size 64 \
  --d_model 64 \
  --learning_rate 0.0001 \
  --train_epochs 50 \
  --dropout 0 \
  --layer_dropout 0 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 4 \
  --n_reshape 12 \
  --patch_len 24 \
  --kernel 9 \
  --patience 10 \
  --des 'Exp' \
  --itr 10



python3 -u run.py \
  --is_training 0 \
  --root_path ./data/dataset/ILI/ \
  --data_path national_illness.csv \
  --model_id ILI_12 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 28 \
  --pred_len 12 \
  --label_len 0 \
  --batch_size 64 \
  --d_model 64 \
  --learning_rate 0.0001 \
  --train_epochs 50 \
  --dropout 0 \
  --layer_dropout 0 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 4 \
  --n_reshape 12 \
  --patch_len 24 \
  --kernel 9 \
  --patience 10 \
  --des 'Exp' \
  --itr 10



python3 -u run.py \
  --is_training 0 \
  --root_path ./data/dataset/ILI/ \
  --data_path national_illness.csv \
  --model_id ILI_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 56 \
  --pred_len 24 \
  --label_len 0 \
  --batch_size 64 \
  --d_model 64 \
  --learning_rate 0.0001 \
  --train_epochs 50 \
  --dropout 0 \
  --layer_dropout 0 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 4 \
  --n_reshape 12 \
  --patch_len 24 \
  --kernel 9 \
  --patience 10 \
  --des 'Exp' \
  --itr 10
