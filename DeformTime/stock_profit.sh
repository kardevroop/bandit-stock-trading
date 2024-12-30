#!/bin/bash -l
#SBATCH -J deform_loss
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/loss/global/noExtra/test_M_180.o
#SBATCH -e ./logs/loss/global/noExtra/test_M_180.e
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
 --model_id SP500aT_1 \
 --model DeformTime \
 --data SP500a  \
 --data_path stocks/all.csv \
 --features MS \
 --target RET \
 --seq_len 180 \
 --pred_len 1 \
 --label_len 0  \
 --batch_size 32 \
 --d_model 32 \
 --learning_rate 0.001 \
 --train_epochs 100 \
 --dropout 0 \
 --layer_dropout 0 \
 --e_layers 2 \
 --d_layers 2 \
 --enc_in 550 \
 --dec_in 550 \
 --c_out 550 \
 --n_heads 4 \
 --n_reshape 24 \
 --patch_len 6 \
 --kernel 7 \
 --patience 20 \
 --des 'Exp' \
 --loss 'Stock_global' \
 --itr 5 \
 --strategy neural \
 --oracle neural \
 --stock_name all \
 --money_pool 1000.0 \
 --valuation \
 --stock_output