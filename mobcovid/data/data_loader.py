import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')



class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len_1 = 24 * 4 * 4
            self.seq_len_2 = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len_1 = size[0]
            self.seq_len_2 = size[1]
            self.label_len = size[2]
            self.pred_len = size[3]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.case = data_path.split(".")[0]
        # self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.3)
        num_vali = len(df_raw) - num_train - num_test

        if self.seq_len_1 >= self.seq_len_2:
            border1s = [0, num_train - self.seq_len_1, len(df_raw) - num_test - self.seq_len_1]
        else:
            border1s = [0, num_train - self.seq_len_2, len(df_raw) - num_test - self.seq_len_2]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        df_data = df_raw[self.target]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler1.fit(train_data.values)

            df_data = self.scaler1.transform(df_data.values).reshape(len(df_data), 2)
            data_1 = df_data[:, 0].reshape(len(df_data), 1)
            data_2 = df_data[:, 1].reshape(len(df_data), 1)

        else:
            data_1 = df_data_1.values.reshape(len(df_data_1), 1)
            data_2 = df_data_2.values.reshape(len(df_data_2), 1)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        data_emergency_stamp = self.emergency_feature(df_stamp)

        self.data_x_1 = data_1[border1:border2]
        self.data_x_2 = data_2[border1:border2]
        if self.inverse:
            self.data_y_1 = df_data_1.values[border1:border2]
            self.data_y_2 = df_data_2.values[border1:border2]
        else:
            self.data_y_1 = data_1[border1:border2]
            self.data_y_2 = data_2[border1:border2]
        self.data_stamp = data_stamp
        self.data_emergency_stamp = data_emergency_stamp

    def __getitem__(self, index):
        if self.seq_len_1 >= self.seq_len_2:
            s_begin_1 = index
            s_end = s_begin_1 + self.seq_len_1
            s_begin_2 = s_end - self.seq_len_2
            seq_x_1 = self.data_x_1[s_begin_1:s_end]
            seq_x_2 = self.data_x_2[s_begin_2:s_end]
            seq_x_mark_1 = self.data_stamp[s_begin_1:s_end]
            seq_x_mark_2 = self.data_stamp[s_begin_2:s_end]
            seq_x_emergency_mark_1 = self.data_emergency_stamp[s_begin_1:s_end]
            seq_x_emergency_mark_2 = self.data_emergency_stamp[s_begin_2:s_end]
        else:
            s_begin_1 = index
            s_end = s_begin_1 + self.seq_len_2
            s_begin_2 = s_end - self.seq_len_1
            seq_x_1 = self.data_x_1[s_begin_2:s_end]
            seq_x_2 = self.data_x_2[s_begin_1:s_end]
            seq_x_mark_1 = self.data_stamp[s_begin_2:s_end]
            seq_x_mark_2 = self.data_stamp[s_begin_1:s_end]
            seq_x_emergency_mark_1 = self.data_emergency_stamp[s_begin_2:s_end]
            seq_x_emergency_mark_2 = self.data_emergency_stamp[s_begin_1:s_end]
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len


        seq_y_1 = self.data_y_1[r_begin:r_end]
        seq_y_2 = self.data_y_2[r_begin:r_end]

        seq_y_mark = self.data_stamp[r_begin:r_end]


        seq_y_emergency_mark = self.data_emergency_stamp[r_begin:r_end]


        return seq_x_1, seq_x_2, seq_y_1, seq_y_2, seq_x_mark_1, seq_x_mark_2, seq_y_mark, seq_x_emergency_mark_1, seq_x_emergency_mark_2, seq_y_emergency_mark

    def __len__(self):
        if self.seq_len_1 >= self.seq_len_2:
            return len(self.data_x_1) - self.seq_len_1 - self.pred_len + 1
        else:
            return len(self.data_x_1) - self.seq_len_2 - self.pred_len + 1

    def inverse_transform1(self, data):
        return self.scaler1.inverse_transform(data)

    def inverse_transform2(self, data):
        return self.scaler2.inverse_transform(data)

    def emergency_function(self, date):
        # Mark two periods
        if self.case == "Tokyo":
            if date.date() >= datetime(2020, 4, 7).date() and date.date() < datetime(2020, 5, 25).date():
                return 1, 0
            elif date.date() >= datetime(2020, 8, 3).date() and date.date() < datetime(2020, 9, 15).date():
                return 0, 1
            elif date.date() >= datetime(2020, 11, 28).date() and date.date() < datetime(2021, 1, 8).date():
                return 0, 1
            elif date.date() >= datetime(2021, 1, 8).date() and date.date() < datetime(2021, 3, 21).date():
                return 1, 1
            elif date.date() >= datetime(2021, 3, 21).date() and date.date() < datetime(2021, 4, 25).date():
                return 0, 1
            elif date.date() >= datetime(2021, 4, 25).date():
                return 1, 1
            else:
                return 0, 0
        elif self.case == "Osaka":
            if date.date() >= datetime(2020, 10, 1).date() and date.date() < datetime(2020, 12, 4).date():
                return 1, 0
            elif date.date() >= datetime(2020, 12, 4).date() and date.date() < datetime(2020, 12, 12).date():
                return 1, 1
            elif date.date() >= datetime(2020, 12, 12).date() and date.date() < datetime(2021, 3, 1).date():
                return 0, 1
            elif date.date() >= datetime(2021, 3, 1).date() and date.date() < datetime(2021, 4, 5).date():
                return 1, 0
            elif date.date() >= datetime(2021, 4, 5).date() and date.date() < datetime(2021, 4, 26).date():
                return 1, 0
            elif date.date() >= datetime(2021, 4, 26).date() and date.date() < datetime(2021, 5, 6).date():
                return 1, 1
            elif date.date() >= datetime(2021, 5, 6).date() and date.date() < datetime(2021, 6, 21).date():
                return 0, 1
            elif date.date() >= datetime(2021, 6, 21).date() and date.date() <= datetime(2021, 6, 28).date():
                return 1, 0
            else:
                return 0, 0

    def emergency_feature(self, dates):
        date = pd.to_datetime(dates.date.values)
        emergency_stamp_1 = pd.Series(date).apply(self.emergency_function)
        if self.case == "Tokyo":
            emergency_stamp = np.zeros((len(emergency_stamp_1), 2))
            for i in range(len(emergency_stamp_1)):
                emergency_stamp[i, 0] = emergency_stamp_1.at[i][0]
                emergency_stamp[i, 1] = emergency_stamp_1.at[i][1]
        elif self.case == "Osaka":
            emergency_stamp = np.zeros((len(emergency_stamp_1), 2))
            for i in range(len(emergency_stamp_1)):
                emergency_stamp[i, 0] = emergency_stamp_1.at[i][0]
                emergency_stamp[i, 1] = emergency_stamp_1.at[i][1]
        return emergency_stamp

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
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
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        # pred_dates = pd.date_range(df.date.values[5], periods=180, freq=self.freq)
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])
        data_emergency_stamp = emergency_feature(df_stamp)



        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_emergency_stamp = data_emergency_stamp
        self.df_stamp = df_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end] # seq_len
        seq_y = self.data_y[r_begin:r_begin+self.label_len] #label_len
        seq_x_mark = self.data_stamp[s_begin:s_end] # seq_len
        seq_y_mark = self.data_stamp[r_begin:r_end] # self.label_len + self.pred_len

        seq_x_emergency_mark = self.data_emergency_stamp[s_begin:s_end]
        seq_y_emergency_mark = self.data_emergency_stamp[r_begin:r_end]
        time_range = pd.to_datetime(self.df_stamp.loc[r_begin:r_end, "date"]).unique().tolist()

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_emergency_mark, seq_y_emergency_mark, time_range
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
