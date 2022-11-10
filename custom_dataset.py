# -*- coding: utf-8 -*-
'''
@Project : min
@File : custom_dataset.py
@Author : Shang Chencheng
@Date : 2022/8/22 13:14
'''
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def collate_fn(batch):
    stock_ID = []
    date = []
    feature = []
    label = []
    for sample in batch:
        stock_ID.extend(sample["stock_ID"])
        date.extend(sample["date"])
        feature.extend(sample["feature"])
        label.extend(sample["label"])
    return torch.FloatTensor(np.stack(feature, axis=0)), torch.FloatTensor(np.stack(label, axis=0)), stock_ID, date


class MinDataset(Dataset):
    def __init__(self, data_dir, tasks, feature_len, look_back, df_r):
        self.data_dir = data_dir
        self.tasks = tasks
        self.feature_len = feature_len + 2
        self.look_back = look_back
        self.df_r = df_r

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        path = []
        for i in task["feature"]:
            path.extend([self.data_dir + str(i)[:10].replace("-", "") + "_500.feather"])

        df_temp = []
        for i in path:
            df = read_df(i)
            d_r = df[['HTSCSecurityID', 'ClosePx', 'OpenPx', 'HighPx', 'LowPx', ]].groupby("HTSCSecurityID").transform(
                lambda x: x.pct_change())
            d_r.columns = ['ClosePx_r', 'OpenPx_r', 'HighPx_r', 'LowPx_r']
            d_r2 = d_r ** 2
            d_r2.columns = ['ClosePx_r2', 'OpenPx_r2', 'HighPx_r2', 'LowPx_r2']
            d_p = df[['HTSCSecurityID', 'NumTrades', 'TotalVolumeTrade', 'TotalValueTrade', ]].groupby(
                "HTSCSecurityID").transform(lambda x: x / x.sum(0))
            d_p.columns = ['NumTrades_p', 'TotalVolumeTrade_p', 'TotalValueTrade_p', ]

            df = pd.concat([df, d_r, d_r2, d_p], axis=1)
            df["posi"] = df['ClosePx_r'] > 0
            df["nega"] = df['ClosePx_r'] < 0
            df = df.dropna(axis=0, how='any')
            df = df.drop(df[df.MDTime == '093000000'].index)
            df["posi"] = df["posi"].astype(int)
            df["nega"] = df["nega"].astype(int)
            df_temp.append(df)
        df = pd.concat(df_temp, axis=0)
        df = df[
            df.groupby(["HTSCSecurityID"]).transform(lambda x: True if len(x) == 240 * self.look_back else False).all(
                axis=1)]
        df.reset_index(drop=True).to_feather("./kline_1/" + str(task["name"])[:10].replace("-", "") + ".feather")

        #        df = pd.read_feather("./kline/" + str(task["name"])[:10].replace("-", "") + ".feather")
        df = df.groupby("HTSCSecurityID").apply(lambda x: x[
            ["OpenPx", "ClosePx", "HighPx", "LowPx", "NumTrades", "TotalValueTrade", "TotalVolumeTrade", 'ClosePx_r',
             'OpenPx_r', 'HighPx_r', 'LowPx_r', 'ClosePx_r2', 'OpenPx_r2', 'HighPx_r2', 'LowPx_r2', 'NumTrades_p',
             'TotalVolumeTrade_p', 'TotalValueTrade_p', 'posi', 'nega']].values)
        # return df
        return {
            "stock_ID": list(df.index),
            "date": [str(task["name"])[:10] for i in df.index],
            "feature": np.stack(list(df)).reshape(-1, self.look_back, 240, self.feature_len),
            "label": self.df_r.loc[str(task["name"])[:10], df.index].values,
        }


def read_df(path):
    return pd.read_feather(path)


class MinFeatDataset(Dataset):
    def __init__(self, feat, y, I, d):
        self.feat = feat
        self.y = y
        self.I = I
        self.d = d

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.feat[idx], self.y[idx], self.I[idx], self.d[idx]



