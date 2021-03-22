import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class TimeDataset(Dataset):
    def __init__(self, root_path, seq_len=24, pred_len=8, split='train', data_path='Demanda_2015.csv',
                 target='demand', scaler=None, step=1, has_minute=False, has_hour=True, use_rolling=False, rolling=8):

        assert split in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[split]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step = step
        self.has_minute = has_minute
        self.has_hour = has_hour
        self.target = target
        self.scaler = scaler
        self.use_rolling = use_rolling
        self.rolling = rolling

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        
        if self.data_path == 'Demanda_2015.csv':
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
            df_raw.columns = ['date', 'time', 'demand']
            df_raw['datetime'] = pd.to_datetime(df_raw['date'].astype(str) + " " + df_raw['time'].astype(str),
                                                format='%d/%m/%Y %H:%M')
            df_raw.drop(['date', 'time'], inplace=True, axis=1)
        elif self.data_path == 'Consumo_uruguay.csv':
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path), sep=';')
            df_raw['holiday'] = (df_raw['holiday']=='feriado').astype(int)
            lbl_encoder = LabelEncoder()
            df_raw['month'] = lbl_encoder.fit_transform(df_raw['month'])
            df_raw['weekday'] = lbl_encoder.fit_transform(df_raw['weekday'])
        elif self.data_path == 'stocks.csv':
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
            df_raw['datetime'] = pd.to_datetime(df_raw['Date'],
                                                format='%Y-%m-%d')
        elif 'train_ML_IOT' in self.data_path:
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
            df_raw['datetime'] = pd.to_datetime(df_raw['DateTime'],
                                                format='%Y-%m-%d %H:%M:%S')
        elif 'household_power_consumption' in self.data_path:

            df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path), sep=';', na_values=['?'])#.iloc[:365*24*60,:]
            df_raw = df_raw.fillna(0)
            df_raw['datetime'] = pd.to_datetime(df_raw['Date'].astype(str) + " " + df_raw['Time'].astype(str),
                                                format='%d/%m/%Y %H:%M:%S')

        elif 'ETT' in self.data_path:
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
            df_raw['datetime'] = pd.to_datetime(df_raw['date'],
                                                format='%Y-%m-%d %H:%M:%S')
        elif 'wine' in self.data_path:
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
                                                                                
        elif self.data_path == 'torneo.csv' or self.data_path == 'aljarafe.csv' or self.data_path == 'bermejales.csv' or self.data_path == 'asomadilla.csv':
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
            df_raw['datetime'] = pd.to_datetime(df_raw['datetime'],format='%d/%m/%Y %H:%M:%S')

        train_size = int(len(df_raw)*0.7)
        valid_size = int(len(df_raw)*0.2)
        test_size = int(len(df_raw)*0.1)


        border1s = [0, train_size, train_size+valid_size]
        border2s = [train_size, train_size+valid_size, train_size+valid_size+test_size]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        
        if 'datetime' in df_raw.columns:
            if self.has_minute:
                border1 += 7 - df_raw.iloc[border1,:].datetime.minute//10 if (df_raw.iloc[border1,:].datetime.minute//10 > 1) else 0 # Set the start in the begining of the minute
            else:
                border1 += 24 - df_raw.iloc[border1,:].datetime.hour if (df_raw.iloc[border1,:].datetime.hour > 1) else 0 # Set the start in the begining of the hour
            border1 = int(border1)
            df_stamp = df_raw[['datetime']][border1:border2]
            df_stamp['year'] = df_stamp.datetime.apply(lambda row: row.year, 1)
            df_stamp['month'] = df_stamp.datetime.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.datetime.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.datetime.apply(lambda row: row.weekday(), 1)
            if self.has_hour:
                df_stamp['hour'] = df_stamp.datetime.apply(lambda row: row.hour, 1)
            if self.has_minute:
                df_stamp['minute'] = df_stamp.datetime.apply(lambda row: row.minute//10, 1)
            
            data_stamp = df_stamp.drop('datetime', axis=1).values
            df_raw = df_raw.drop('datetime', axis=1)
        elif 'year' in df_raw.columns:
            border1 += 25 - df_raw.iloc[border1,:].hour if (df_raw.iloc[border1,:].hour > 1) else 0 # Set the start in the begining of the day
            border1 = int(border1)
            df_stamp = df_raw[border1:border2].loc[:, ['year', 'month', 'day', 'weekday', 'hour']]
            df_raw = df_raw.drop(['year', 'month', 'day', 'weekday', 'hour'], axis=1)
            
            data_stamp = df_stamp.values
            
            if self.has_minute:
                df_stamp['minute'] = df_stamp.minute
                df_raw = df_raw.drop('minute', axis=1)
        else:
            df_stamp = None

        if self.use_rolling:
            df_raw['rolling_target'] = df_raw[[self.target]].rolling(window=self.rolling).mean().fillna(method='backfill')

        if self.scaler is None:
            self.scaler = StandardScaler()
            if self.use_rolling:
                self.scaler.fit(df_raw[[self.target, 'rolling_target']].values)
            else:    
                self.scaler.fit(df_raw.values)
        
        if self.use_rolling:
            data = self.scaler.transform(df_raw[[self.target, 'rolling_target']].values)
        else:
            data = self.scaler.transform(df_raw.values)
        
        self.data_y_raw = df_raw[[self.target]][border1:border2].values
        
        mean, std = self.target_scale_factors()
        self.data_y = (self.data_y_raw - mean)/std
        
        self.data_x = data[border1:border2]
        self.data_stamp = data_stamp
        
    def target_scale_factors(self):
        
        return self.data_y_raw.mean(), self.data_y_raw.std()

    def __getitem__(self, index):
        s_begin = index*self.step
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        if self.data_stamp is not None:
            seq_x_mark = self.data_stamp[s_begin:s_end,1:]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        else:
            seq_x_mark = None
            seq_y_mark = None

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)//self.step - self.seq_len - self.pred_len + 1
        
class UCRDataset(Dataset):
    def __init__(self, root_path, split='train', data_path='wine.csv',
                 target=0, scaler=None, use_rolling=False, rolling=8):

        assert split in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[split]
        self.target = target
        self.scaler = scaler
        self.use_rolling = use_rolling
        self.rolling = rolling

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path)).drop('Unnamed: 0', axis=1).sample(frac=1, random_state=123).reset_index(drop=True)
                                                                              
        train_size = int(len(df_raw)*0.5)
        valid_size = int(len(df_raw)*0.2)
        test_size = int(len(df_raw)*0.3)


        border1s = [0, train_size, train_size+valid_size]
        border2s = [train_size, train_size+valid_size, train_size+valid_size+test_size]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(df_raw.drop(columns=df_raw.columns[self.target]).values)

        data = self.scaler.transform(df_raw.drop(columns=df_raw.columns[self.target]).values)
        
        self.data_y = df_raw.iloc[:, self.target][border1:border2].values - 1
        self.data_x = data[border1:border2]
        
    def __getitem__(self, index):
        X = np.expand_dims(self.data_x[index], axis=1)
        if self.use_rolling:
            roll = np.expand_dims(np.convolve(X, np.ones(self.rolling), mode='same')/len(X), axis=1)
            X = np.concatenate((X, roll), axis=1)
            
        return X, self.data_y[index]

    def __len__(self):
        return len(self.data_x)
