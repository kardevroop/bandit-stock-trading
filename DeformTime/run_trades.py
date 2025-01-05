import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from src.exp.exp_MTS_forecasting import exp_MTS_forecasting
from src.utils.tools import seed_everything
import sys, os

# 
torch.set_num_threads(4)

if __name__ == '__main__':
    seed_everything(seed=2021)

    parser = argparse.ArgumentParser(description='MTS forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='DeformTime',
                        help='model name, options: [DLinear, LightTS, PatchTST, Crossformer, iTransformer, DeformTime]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--ticker_file', type=str, default='tickers.csv', help='ticker file, needed if data is SP500')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    #parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')

    parser.add_argument('--target', nargs='+', default=['OT'], help='target feature in S or MS task')

    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--patch_len', type=int, default=7, help='patch length of each split')
    parser.add_argument('--stride', type=int, default=4, help='stride when splitting')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--layer_dropout', type=float, default=0.6, help='path   dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--gamma', type=float, default=10, help='gamma value for smooth losses')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # DeformTime parameters
    parser.add_argument('--kernel', type=int, default=6, help='kernel size')
    parser.add_argument('--n_reshape', type=int, default=16)

    # Trading Strategy parameters
    parser.add_argument('--strategy', type=str, default='bandit', help="Select Trading startegy")
    parser.add_argument('--oracle', type=str, default='portfolio', help="Select Oracle type for reward calculation")
    parser.add_argument('--money_pool', type=float, default=1000.0, help="The money to start trading with on day 1")
    parser.add_argument('--stock_name', type=str, default='AKAM', help="Name of Stock to read")
    parser.add_argument('--valuation', action='store_true', help="Set flag to calculate final profit")
    parser.add_argument('--enable_action', action='store_true', help="Set flag to enable extra output node to do nothing")
    parser.add_argument('--stock_output', action='store_true', help="Set flag to change final layer")
    # parser.add_argument('--v_dim', type='int', default=50, help="Number of V outputs")

    args = parser.parse_args()
    args.stride = args.patch_len
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print(args.target)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    Exp = exp_MTS_forecasting

    mses, maes, times = [], [], []
    total_time = 0

    if args.is_training:
        for ii in range(args.itr):
            print(f'Run {ii+1}')
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_d{}_ld{}_lr{}_loss{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.dropout,
                args.layer_dropout,
                args.learning_rate,
                args.loss,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            model, time_taken, scaler = exp.train(setting)
            total_time += time_taken
            print(f'[INFO    ] Number of parameters: {exp._model_parameter_count(model)}')
            print(f'[INFO    ] Total time taken: {total_time}')

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, mse = exp.test(setting, scaler=scaler)
            torch.cuda.empty_cache()

    else:
        best_metric, best_model = sys.maxsize, None
        for ii in range(args.itr):
            #ii = 0
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_d{}_ld{}_lr{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.dropout,
                args.layer_dropout,
                args.learning_rate,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>validation : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, mse = exp.vali(None, None, None, setting=setting, val=1)
            if mae < best_metric:
                best_metric = mae
                best_model = setting
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(best_model))
        exp.test(best_model, test=1)
        torch.cuda.empty_cache()
