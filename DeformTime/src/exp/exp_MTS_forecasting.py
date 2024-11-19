from data.data_provider.data_factory import data_provider
from src.exp.exp_basic import Exp_Basic
from src.utils.tools import EarlyStopping, adjust_learning_rate
from src.utils.metrics import MAE, MSE, SMAPE
from src.utils.losses import stock_loss
from strategy.strategy import Strategy
from oracle.oracle import Oracle
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class exp_MTS_forecasting(Exp_Basic):
    def __init__(self, args):
        super(exp_MTS_forecasting, self).__init__(args)

    def _build_model(self):
        # print(self.args.v_dim)
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, scaler=None):
        data_set, data_loader = data_provider(self.args, flag, scaler=scaler)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_type):
        if loss_type == 'MAE':
            criterion = nn.L1Loss()
        if loss_type == 'Stock':
            criterion = stock_loss(self.args.enable_action)
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, setting=None, val=0):
        total_loss = []
        train_mae = []

        preds = []
        trues = []

        if vali_data is None:
            vali_data, vali_loader = self._get_data(flag='val')
        if val:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_next, batch_y_next) in enumerate(vali_loader):

                dim_len = vali_data.target.shape[1]  + (1 if self.args.enable_action else 0)

                f_dim = -1 * dim_len if self.args.features == 'MS' else 0
                f_dim += (1 if self.args.enable_action else 0)
                batch_y_next = batch_y[:, -self.args.pred_len:, 3*f_dim:2*f_dim].to(self.device)

                batch_x = torch.dstack((batch_x[:,:,:3*f_dim], batch_x[:,:,f_dim:]))
                batch_y = torch.dstack((batch_y[:,:,:3*f_dim], batch_y[:,:,f_dim:]))
                batch_x_mark = torch.dstack((batch_x_mark[:,:,:3*f_dim], batch_x_mark[:,:,f_dim:]))
                batch_y_mark = torch.dstack((batch_y_mark[:,:,:3*f_dim], batch_y_mark[:,:,f_dim:]))

                f_dim -= (1 if self.args.enable_action else 0)
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # print(f"f_dim: {f_dim}")

                # f_dim -= (1 if self.args.enable_action else 0)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                f_dim += (1 if self.args.enable_action else 0)

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # batch_y_next = batch_y_next[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                true_n = batch_y_next.detach().cpu()

                true = true.float()
                true_n = true_n.float()

                # weights = torch.abs(outputs) / torch.sum(torch.abs(outputs)).item()
                # weights = weights.detach().cpu()

                if criterion is not None:
                    loss = criterion(outputs.detach().cpu(), true, target_next=true_n)
                    #mae = MAE(pred[:,-1,-1], true[:,-1,-1])

                    total_loss.append(loss)
                    #train_mae.append(mae)

                pred = pred.numpy()
                true = true.numpy()

                preds.append(pred)
                trues.append(true)

        total_loss = np.average(total_loss)
        train_mae = np.average(train_mae)

        if val:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mae, mse = MAE(preds[:,-1,:], trues[:,-1,:]), MSE(preds[:,-1,:], trues[:,-1,:])
            print('val mse:{}, val mae:{}'.format(mse, mae))
            return mae, mse

        self.model.train()
        return total_loss, train_mae

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val', scaler=train_data.scaler)
        test_data, test_loader = self._get_data(flag='test', scaler=train_data.scaler)

        self.strategy = Strategy.initialize(self.args)
        self.oracle = Oracle.initialize(self.args)
        self.args.v_dim = len(train_data.stocks) + (1 if self.args.enable_action else 0)
        # print(f"[INFO] v_dim: {self.args.v_dim}")

        self.model = self._build_model().to(self.device)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=1e-16)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_next, batch_y_next) in enumerate(train_loader):
                # if i == len(train_data) - 1:
                #     continue
                iter_count += 1
                model_optim.zero_grad()

                dim_len = train_data.target.shape[1] + (1 if self.args.enable_action else 0)
                f_dim = -1 * dim_len if self.args.features == 'MS' else 0
                f_dim += (1 if self.args.enable_action else 0)
                batch_y_next = batch_y[:, -self.args.pred_len:, 3*f_dim:2*f_dim].to(self.device)

                batch_x = torch.dstack((batch_x[:,:,:3*f_dim], batch_x[:,:,f_dim:]))
                batch_y = torch.dstack((batch_y[:,:,:3*f_dim], batch_y[:,:,f_dim:]))
                batch_x_mark = torch.dstack((batch_x_mark[:,:,:3*f_dim], batch_x_mark[:,:,f_dim:]))
                batch_y_mark = torch.dstack((batch_y_mark[:,:,:3*f_dim], batch_y_mark[:,:,f_dim:]))

                f_dim -= (1 if self.args.enable_action else 0)

                # print(f"batch_x shape: {batch_x.shape}")

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # print(f"[INFO    ]       x_mark:\n {batch_x_mark}")

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # print(f"[INFO]   Dimension: {dim_len}")

                        # f_dim = -1 * dim_len if self.args.features == 'MS' else 0

                        #f_dim -= (1 if self.args.enable_action else 0)
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        f_dim += (1 if self.args.enable_action else 0)

                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        # batch_y_next = batch_y_next[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # batch_y_next = batch_y[:, -self.args.pred_len:, 3*f_dim:2*f_dim].to(self.device)

                        # print(f"y shape: {batch_y.shape}")
                        # print(f"y_next shape: {batch_y_next.shape}")

                        # print(f"[INFO]   Output Shape: {outputs.shape}")

                        # weights = torch.abs(outputs) / torch.sum(torch.abs(outputs)).item()

                        loss = criterion(outputs, batch_y, target_next=batch_y_next)
                        train_loss.append(loss.item())

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # dim_len = train_data.target.shape[1] + (1 if self.args.enable_action else 0)

                    # print(f"[INFO]   Dimension: {dim_len} | {train_data.target.shape[1]}")

                    # f_dim = -1 * dim_len if self.args.features == 'MS' else 0 # Only use the target variable


                    # f_dim -= (1 if self.args.enable_action else 0)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    f_dim += (1 if self.args.enable_action else 0)

                    # print(f"[INFO]   f_dim: {f_dim}")

                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # batch_y_next = batch_y[:, -self.args.pred_len:, 3*f_dim:2*f_dim].to(self.device)


                    batch_y = batch_y.float()
                    batch_y_next = batch_y_next.float()

                    # print(f"y shape: {batch_y.shape}")
                    # print(f"y_next shape: {batch_y_next.shape}")

                    # print(f"[INFO]   Output Shape: {outputs.shape}")

                    # weights = torch.abs(outputs) / torch.sum(torch.abs(outputs)).item()

                    loss = criterion(outputs, batch_y, target_next=batch_y_next)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            time_taken = time.time() - epoch_time
            train_loss = np.average(train_loss)
            vali_loss, vali_mse = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mse = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} Vali MSE: {5:.7f} Test MSE: {6:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, vali_mse, test_mse))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print(f"Early stopping on validation score {early_stopping.best_score}")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, time_taken, train_data.scaler

    def test(self, setting, test=0, scaler=None):
        test_data, test_loader = self._get_data(flag='test', scaler=scaler)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        net_profit, money_per_day = 0.0, 0.0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_next, batch_y_next) in enumerate(test_loader):
                print(f"[INFO]   Test {i}")

                # print(f"test target len: {len(test_data.target)}")
                dim_len = test_data.target.shape[1] + (1 if self.args.enable_action else 0)
                f_dim = -1 * dim_len if self.args.features == 'MS' else 0
                f_dim += (1 if self.args.enable_action else 0)
                batch_y_next = batch_y[:, -self.args.pred_len:, 3*f_dim:2*f_dim].to(self.device)
                
                batch_x1 = torch.dstack((batch_x[:,:,:3*f_dim], batch_x[:,:,f_dim:]))
                batch_y1 = torch.dstack((batch_y[:,:,:3*f_dim], batch_y[:,:,f_dim:]))
                batch_x_mark1 = torch.dstack((batch_x_mark[:,:,:3*f_dim], batch_x_mark[:,:,f_dim:]))
                batch_y_mark1 = torch.dstack((batch_y_mark[:,:,:3*f_dim], batch_y_mark[:,:,f_dim:]))

                f_dim -= (1 if self.args.enable_action else 0)
                
                batch_x1 = batch_x1.float().to(self.device)
                batch_y1 = batch_y1.float().to(self.device)
                batch_x_mark1 = batch_x_mark1.float().to(self.device)
                batch_y_mark1 = batch_y_mark1.float().to(self.device)

                # print(f"batch_x: {batch_x.shape}")

                # decoder input
                dec_inp = torch.zeros_like(batch_y1[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y1[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x1, batch_x_mark1, dec_inp, batch_y_mark1)[0]
                        else:
                            outputs = self.model(batch_x1, batch_x_mark1, dec_inp, batch_y_mark1)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x1, batch_x_mark1, dec_inp, batch_y_mark1)[0]

                    else:
                        outputs = self.model(batch_x1, batch_x_mark1, dec_inp, batch_y_mark1)

                batch_x_next, _, _, _, _, _ = test_data[i+1]
                batch_x_next = np.expand_dims(batch_x_next, axis=0)

                # print(f"batch_x_next: {batch_x_next.shape}")
                batch_x_next = torch.from_numpy(batch_x_next)

                f_dim += (1 if self.args.enable_action else 0)

                batch_x_next1 = torch.dstack((batch_x_next[:,:,:3*f_dim], batch_x_next[:,:,f_dim:])).float().to(self.device)
                # print(f"batch_x1: {batch_x1.shape}")
                # print(f"batch_x_next1: {batch_x_next1.shape}")
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                batch_y1 = batch_y1[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y1 = batch_y1.detach().cpu().numpy()
                batch_y_next = batch_y_next.detach().cpu().numpy()

                # print(f"[INFO    ]       test outputs: {outputs.shape}")
                 
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    # outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                batch_x = batch_x.detach().cpu().numpy()
                batch_x_next = batch_x_next.detach().cpu().numpy()
                shape = batch_x.shape

                batch_x = test_data.inverse_transform(batch_x.squeeze(0))
                batch_x_next = test_data.inverse_transform(batch_x_next.squeeze(0))

                # print(f"batch_x: {batch_x.shape}")
                # print(f"batch_x_next: {batch_x_next.shape}")

                context = pd.DataFrame(batch_x, columns=test_data.cols)
                context_next = pd.DataFrame(batch_x_next, columns=test_data.cols)

                # print(f"[INFO    ]       context: {context.tail()}")
                # print(f"[INFO    ]       context_next: {context_next.tail()}")

                if self.args.valuation:
                    n = context.shape[0]
                    # net_reward = 0.0
                    # for index in range(context.shape[0]-1):
                    # print(context.shape)
                    # print(context.iloc[index].to_dict().keys())
                    self.strategy.make_move(context=context.iloc[-1,:].to_dict(), 
                                            forecast=context_next.iloc[0,:].to_dict(), 
                                            decision=torch.tensor(outputs).squeeze(0), 
                                            stocks=test_data.stocks + [None])
                    reward = self.oracle.calculate_reward(state=self.strategy.get_state(), context=context_next.iloc[0,:].to_dict())
                    # net_reward += reward
                    net_profit += reward
                    money_per_day += self.strategy.money_pool
                    self.strategy.report_reward(reward)
                    self.oracle.reset()
                    self.strategy.reset()
                    # net_profit += net_reward / n
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        # preds = np.array(preds)
        # trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        print(f"[INFO    ]       Profit: {net_profit} | {net_profit * 100.0 / (money_per_day)}%")

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        target_mae, target_mse = 0.0, 0.0
        # target_mae = MAE(preds[:,-1,:], trues[:,-1,:])
        # target_mse = MSE(preds[:,-1,:], trues[:,-1,:])
        # target_smape = SMAPE(preds[:,-1,:], trues[:,-1,:])
        # print("target mae:{}".format(target_mae))
        # print("target mse:{}".format(target_mse))
        # print("target smape:{}".format(target_smape))

        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mae:{}, smape:{}'.format(target_mae, target_smape))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'pred.npy', preds)
        # np.savetxt(folder_path + 'pred.txt', preds[:,:,0])
        # np.save(folder_path + 'true.npy', trues)

        return target_mae, target_mse

    def _model_parameter_count(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
