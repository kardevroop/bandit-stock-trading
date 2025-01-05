import os
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from src.utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

def corr_with(trends, keys):
    # Correlate by a specific period
    trends_ = copy.deepcopy(trends)
    trends_selected = trends_[keys]
    try:
        correlation_score = trends_.corrwith(trends_selected, axis=0, numeric_only=True)
    except:
        correlation_score = trends_.corrwith(trends_selected, axis=0)
    corr_filtered = pd.DataFrame(correlation_score).fillna(0)
    corr_filtered = corr_filtered.reset_index()
    
    sorted_correlation = correlation_score.sort_values(ascending=True)
    return corr_filtered, sorted_correlation


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target=['OT'], scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns.values) #Make a list of all of the columns in the df
        for output_col in self.target:
            cols.pop(cols.index(output_col)) #Remove target col from list
        df_raw = df_raw[cols+self.target]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            # select the data from train base on the correlation
            train_data = df_data[border1s[0]:border2s[0]]
            corr_rate, sorted_corr_rate = corr_with(train_data, 'OT')
            selected_columns = sorted_corr_rate[abs(sorted_corr_rate)>=0].index
            # self.dim = len(selected_columns)
            selected_columns = cols_data
            df_data = df_data[selected_columns]
        elif self.features == 'S':
            df_data = df_raw[self.target]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            # select the data from train base on the correlation
            train_data = df_data[border1s[0]:border2s[0]]
            corr_rate, sorted_corr_rate = corr_with(train_data, 'OT')
            selected_columns = sorted_corr_rate[abs(sorted_corr_rate)>=0].index
            # self.dim = len(selected_columns)
            df_data = df_data[selected_columns]
        elif self.features == 'S':
            df_data = df_raw[self.target]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            # select the data from train base on the correlation
            train_data = df_data[border1s[0]:border2s[0]]
            corr_rate, sorted_corr_rate = corr_with(train_data, 'OT')
            selected_columns = sorted_corr_rate[abs(sorted_corr_rate)>=0].index
            # self.dim = len(selected_columns)
            df_data = df_data[selected_columns]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_SP500(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='stocks/AKAM.csv',
                 target=['OT'], scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        cols = list(df_raw.columns)
        for output_col in self.target:
            cols.pop(cols.index(output_col)) #Remove target col from list
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + self.target]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.15)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            # select the data from train base on the correlation
            train_data = df_data[border1s[0]:border2s[0]]
            corr_rate, sorted_corr_rate = corr_with(train_data, self.target[-1])
            selected_columns = sorted_corr_rate[abs(sorted_corr_rate)>=0].index
            # self.dim = len(selected_columns)
            df_data = df_data[selected_columns]
        elif self.features == 'S':
            df_data = df_raw[self.target]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_SP500_all(Dataset):
    '''
    Dataset for reading stock information. Currently reads only one stock

    Args
    ----
    data_path: str
        Path to base folder containing datasets
    dataset: str
        The dataset folder to load
    split: str
        Whether the task is to train, validate or test
    stock: str
        The specific stock file to read
    target: str
        The target column to predict
    transform
        Pytorch transformations on data
    scale
        Whether to normalize the data
    '''
    def __init__(self,
                root_path,
                flag="train",
                size=None,
                features='M',
                data_path='stocks/all.csv',
                column_names=None,
                target=['RET'],
                transform=None,
                scale=True, 
                timeenc=0, 
                freq='h', 
                seasonal_patterns=None,
                target_subset=True,
                scaler=None):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.flag=flag
        self.set_type = type_map[flag]

        self.features = features
        self.target_column = target[0]
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.target_subset = target_subset
        self.scaler = scaler

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        if self.scaler is None:
            self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        self.target = list(df_raw.filter(regex=f".*{self.target_column}"))

        tmp = []
        for t in self.target:
            if "SELL" not in t:
                tmp.append(t)
        self.target = [a for a in tmp]

        for t in self.target:
            cols.remove(t)
        self.target = list(filter(lambda x: 'expected' not in x and 'predicted' not in x, self.target))

        self.target_next = list(df_raw.filter(regex=f".*expected_{self.target_column}"))
        self.target_pred = list(df_raw.filter(regex=f".*predicted_{self.target_column}"))

        # print(f"before: {self.target_next}\n")
        # print(f"after: {self.target}\n")

        # sell_prc = list(df_raw.filter(regex=f".*_SELL_PRC"))
        # tran = list(df_raw.filter(regex=f".*_TRAN_COST"))
        # for t in sell_prc:
        #     cols.remove(t)
        # for t in tran:
        #     cols.remove(t)

        sell_prc = list(df_raw.filter(regex=f".*_SELL_PRC"))
        for t in sell_prc:
            cols.remove(t)

        tran_cost = list(df_raw.filter(regex=f".*_TRAN_COST"))
        for t in tran_cost:
            cols.remove(t)

        # ask = list(df_raw.filter(regex=f".*_ASK"))
        # for t in ask:
        #     cols.remove(t)

        # shrout = list(df_raw.filter(regex=f".*_SHROUT"))
        # for t in shrout:
        #     cols.remove(t)

        # market_cap = list(df_raw.filter(regex=f".*_MARKET_CAP"))
        # for t in market_cap:
        #     cols.remove(t)

        # bid = list(df_raw.filter(regex=f".*_BID"))
        # for t in bid:
        #     cols.remove(t)

        cols.remove('date')

        df_raw = df_raw[['date'] + cols + self.target_next + self.target_pred + self.target]
        self.df_raw = df_raw.copy()

        self.stocks = [a.split("_")[0] for a in self.target]

        if isinstance(self.target, pd.Series):
            self.target = self.target.to_frame()

        df_raw['date'] = pd.to_datetime(df_raw['date'])

        train_df = df_raw[df_raw['date'].dt.year < 2022]
        val_df = df_raw[df_raw['date'].dt.year == 2022]
        test_df = df_raw[df_raw['date'].dt.year == 2023]

        # print(len(train_df))
        # print(len(val_df))
        # print(len(test_df))

        num_train = len(train_df)
        num_test = len(test_df)
        num_vali = len(df_raw) - num_train - num_test

        # num_train = int(len(df_raw) * 0.8)
        # num_test = int(len(df_raw) * 0.1)
        # num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # print(f"border1s: {border1s}")

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            self.cols = cols_data
            df_data = df_raw[cols_data]
            # select the data from train base on the correlation
            train_data = df_data[border1s[0]:border2s[0]]
            # corr_rate, sorted_corr_rate = corr_with(train_data, self.target[-1])
            # selected_columns = sorted_corr_rate[abs(sorted_corr_rate)>=0].index
            # self.dim = len(selected_columns)
            # df_data = df_data[selected_columns]
        elif self.features == 'S':
            df_data = df_raw[self.target]

        # print(df_data.shape)

        self.df_data_denorm = df_data.copy().values

        self.features = df_raw[cols]
        self.target = df_raw[self.target]

        # print(f"[INFO   ]       {self.flag} shape: {train_data.values.shape}")
        # print(f"[INFO   ]       df_raw :\n {df_raw.head()}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            print(df_stamp.head())
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.df_data = df_data.iloc[border1:border2,:]
        self.data_stamp = data_stamp


    def __len__(self) -> int:
        # return self.features.shape[0]
        return len(self.data_x) - self.seq_len - self.pred_len + 1 # (1 if self.set_type !=2 else 0)

    def __getitem__(self, index) -> any:

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        df_seq_x = self.df_data[s_begin:s_end]
        df_seq_y = self.df_data[r_begin:r_end]

        seq_x_n, seq_y_n, df_seq_x_n, df_seq_y_n = self.df_data_denorm[s_begin:s_end], \
                                                    self.df_data_denorm[s_begin:s_end], \
                                                    self.df_data_denorm[s_begin:s_end], \
                                                    self.df_data_denorm[s_begin:s_end]

        # if index < len(self.data_x) - self.seq_len - self.pred_len:

        #     s_next = index+1
        #     s_nend = s_next + self.seq_len
        #     r_nbegin = s_nend - self.label_len
        #     r_nend = r_nbegin + self.label_len + self.pred_len

        #     seq_x_n = self.data_x[s_next:s_nend]
        #     seq_y_n = self.data_y[r_nbegin:r_nend]
        #     # seq_x_nmark = self.data_stamp[s_next:s_nend]
        #     # seq_y_nmark = self.data_stamp[r_nbegin:r_nend]
        #     df_seq_x_n = self.df_data[s_next:s_nend]
        #     df_seq_y_n = self.df_data[r_nbegin:r_nend]
        # else:
        #     seq_x_n = seq_x.copy()
        #     seq_y_n = seq_y.copy()
        #     df_seq_x_n = df_seq_x.copy()
        #     df_seq_y_n = df_seq_y.copy()

        # print(df_seq_x.to_dict())

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_n, seq_y_n

        # if index == self.features.shape[0] - 1:
        #     return self.feature_tensors[index], \
        #             torch.zeros_like(self.feature_tensors[index]), \
        #             self.features.iloc[index,:].to_dict(), \
        #             self.features.iloc[index,:].to_dict(), \
        #             self.target_tensors[index], \
        #             torch.zeros_like(self.target_tensors[index])

        # return self.feature_tensors[index], \
        #         self.feature_tensors[index+1], \
        #         self.features.iloc[index,:].to_dict(), \
        #         self.features.iloc[index+1,:].to_dict(), \
        #         self.target_tensors[index], \
        #         self.target_tensors[index+1]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
