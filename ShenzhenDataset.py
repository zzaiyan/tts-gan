import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class shenzhen_dataset(Dataset):
    def __init__(self,
                 data_size=-1,  # 启用的数据数量
                 path='./shenzhen/',  # shenzhen文件夹路径
                 train_rate=0.8,  # 训练集占比
                 data_mode='Train',  # 数据模式
                 verbose=False
                 ):
        self.path = path
        self.data_size = data_size
        self.train_rate = train_rate
        self.data_mode = data_mode
        self.verbose = verbose

        if not os.path.exists(self.path):
            print('Path not existing!')
            raise ValueError

        assert self.data_mode == 'Train' or self.data_mode == 'Test'

        # 加载所有文件目录
        areas = os.listdir(self.path)
        if self.verbose:
            print(areas)
        add_ele = []
        for area in areas:
            if area == '.DS_Store':
                continue
            dest = self.path + area
            t_dir = os.listdir(dest)
            # print(t_dir)
            for zone in t_dir:
                if zone == '.DS_Store':
                    continue
                items = os.listdir(dest + '/' + zone)
                for ele in items:
                    if ele == '.DS_Store':
                        continue
                    add_ele.append(f'{dest}/{zone}/{ele}')
        if self.verbose:
            print('Num of elements:', len(add_ele))

        if self.data_size == -1:
            self.data_size = len(add_ele)

        # 读取所有CSV文件
        self.raw_data = []
        looper = zip(range(self.data_size), add_ele)
        if self.verbose:
            looper = tqdm(looper)
        for idx, add in looper:
            # if add.count('.DS_Store'):
            #     continue
            df = pd.read_csv(add, usecols=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'QA'],
                             na_values=[0, -100])
            df.interpolate(method='linear', inplace=True)
            df.fillna(method='pad', inplace=True)
            df.fillna(method='backfill', inplace=True)
            # has_null = df.isna().any().any()
            # if has_null:
            #     print(add)
            #     raise ValueError
            df = df.astype('float')
            self.raw_data.append(df.values.transpose())

        # 打乱原始数据
        random.seed(777)
        random.shuffle(self.raw_data)
        # 转化为array，并reshape
        self.np_data = np.array(self.raw_data)
        self.np_data = self.np_data.reshape((-1, 11, 1, 64))
        # 划分数据集
        self.train_data = self.np_data[:self._train_size()]
        self.test_data = self.np_data[self._train_size():]
        # 正则化
        self.normalization(self.train_data)
        self.normalization(self.test_data)

        if self.data_mode == 'Train':
            self.shape = self.train_data.shape
        if self.data_mode == 'Test':
            self.shape = self.test_data.shape

    def _normalize(self, epoch):
        e = 1e-10
        result = (epoch - epoch.mean(axis=0)) / \
            ((np.sqrt(epoch.var(axis=0))) + e)
        return result

    def normalization(self, epochs):
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
                epochs[i, j, 0, :] = self._normalize(epochs[i, j, 0, :])

    def _train_size(self):
        return int(self.data_size * self.train_rate)

    def __len__(self):
        if self.data_mode == 'Train':
            return len(self.train_data)
        if self.data_mode == 'Test':
            return len(self.test_data)
        raise RuntimeError

    def __getitem__(self, idx):
        if self.data_mode == 'Train':
            return self.train_data[idx], 7
        if self.data_mode == 'Test':
            return self.test_data[idx], 7
        raise RuntimeError
