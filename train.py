# -*- coding: utf-8 -*-
'''
@Project : min
@File : train.py
@Author : Shang Chencheng
@Date : 2022/8/22 13:17
'''
import scipy.io as scio
import os
import copy
import itertools
import logging
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from log_module import get_module_logger

from custom_dataset import *
import torch
import torch.optim as optim
from models import *
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class trainModels:
    def __init__(self, params):
        if params.model_name == 'AlphaNet_min_intraday':
            self.model = AlphaNet_min_intraday(params.feature_num, 30, output_size=1)

        elif params.model_name == 'AlphaNet_min_interday':
            self.model = AlphaNet_min_interday(params.feature_num, 30, output_size=1)
            self.model.GRU.load_state_dict(torch.load(params.model_load_dir + "AlphaNet_min_intraday.pt")['model'])
        else:
            raise Exception("No such a network model.")
        self.logger = get_module_logger(log_file=params.log_name)
        self.epochs = params.epochs
        self.batch_size = params.batch_size
        self.max_patience = params.max_patience
        if torch.cuda.device_count() == 0:
            self.gpu_available = False
        else:
            self.gpu_available = True
            torch.cuda.set_device(params.gpu_id)
            self.model.cuda()
        self.model_save_dir = params.model_save_dir
        self.model_name = params.model_name
        self.loss_fn = torch.nn.MSELoss()
        # self.eval_fn = pearson_R
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=0.00001)

        self.best_model = None

    def reset(self):
        # self.model =
        self.optimizer = optim.Adam(self.model.parameters(), )

    def _train_loop(self, loader, is_train=True, load_flag=True):

        epoch_loss = 0.0
        label = []
        pred = []
        stock_ID = []
        date = []
        feat = []
        # epoch_metric = 0.0
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        for x, y, I, d in tqdm(loader):
            if self.gpu_available:
                x = x.float().cuda()
                y = y.float().cuda()
            if load_flag:
                y_pred, x_feat = self.model(x, load_flag)
            else:
                y_pred = self.model(x, load_flag)
            y_pred = y_pred[~torch.isnan(y)]
            I = np.array(I)[~np.isnan(y.detach().cpu().numpy())]
            d = np.array(d)[~np.isnan(y.detach().cpu().numpy())]
            y = y[~torch.isnan(y)]
            if load_flag:
                x_feat = x_feat[~torch.isnan(y)]

            y_pred = y_pred[~torch.isinf(y)]
            I = np.array(I)[~np.isinf(y.detach().cpu().numpy())]
            d = np.array(d)[~np.isinf(y.detach().cpu().numpy())]
            y = y[~torch.isinf(y)]
            if load_flag:
                x_feat = x_feat[~torch.isinf(y)]

            batch_loss = self.loss_fn(y_pred, y)
            # print(sum(y_pred), sum(y), batch_loss)
            # print(self.eval_fn(y_pred, y))
            pred.extend(list(y_pred.detach().cpu().numpy()))
            label.extend(list(y.cpu().numpy()))
            stock_ID.extend(list(I))
            date.extend(list(d))
            if load_flag:
                feat.append(x_feat.detach().cpu().numpy())
            # batch_metric = self.eval_fn(y_pred, y)
            if is_train:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            epoch_loss += batch_loss.item()

        df_pred, df_label = self.Dense2Square(label, pred, stock_ID, date)
        IC = self.cal_IC(df_pred, df_label)

        if load_flag:
            feat = np.vstack(feat)
            min_feat_dataset = MinFeatDataset(feat, label, stock_ID, date)
            if is_train:
                min_feat_dataloader = DataLoader(min_feat_dataset, self.batch_size, num_workers=1, shuffle=True, )
            else:
                min_feat_dataloader = DataLoader(min_feat_dataset, self.batch_size, num_workers=1, shuffle=False, )

            return epoch_loss / len(loader), IC, min_feat_dataloader
        else:
            return epoch_loss / len(loader), IC

    def train(self, trainloader, evalloader):
        start_time = time.time()
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []
        best_val = -np.inf
        patience_counter = 0

        for i in range(self.epochs):
            if i == 0:
                epoch_loss, epoch_metric, trainloader = self._train_loop(
                    trainloader, is_train=True, load_flag=True)

                train_losses.append(epoch_loss)
                train_metrics.append(epoch_metric)

                epoch_loss, epoch_metric, evalloader = self._train_loop(
                    evalloader, is_train=False, load_flag=True)

                val_losses.append(epoch_loss)
                val_metrics.append(epoch_metric)
            else:
                epoch_loss, epoch_metric = self._train_loop(
                    trainloader, is_train=True, load_flag=False)

                train_losses.append(epoch_loss)
                train_metrics.append(epoch_metric)

                epoch_loss, epoch_metric = self._train_loop(
                    evalloader, is_train=False, load_flag=False)

                val_losses.append(epoch_loss)
                val_metrics.append(epoch_metric)

            if val_metrics[-1] > best_val:
                best_val = val_metrics[-1]
                self.best_model = copy.deepcopy(self.model)
                patience_counter = 0
                if not os.path.exists(self.model_save_dir):
                    # print('create path: %s' % model_path)
                    os.makedirs(self.model_save_dir)

                torch.save({
                    'model': self.model.GRU.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': i,
                }, self.model_save_dir + self.model_name + '.pt')

                self.logger.info(
                    'Save best model as {}.pt'.format(
                        self.model_save_dir +
                        self.model_name))

            else:
                patience_counter += 1
                if patience_counter > self.max_patience:
                    self.logger.info(
                        'Max-patience reached after EPOCH {}'.format(i))
                    break

            self.logger.info("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN METRIC]: %.3f" % (
                i, train_losses[-1], train_metrics[-1]))
            self.logger.info("[EPOCH]: %i, [VAL LOSS]: %.6f, [VAL METRIC]: %.3f" % (
                i, val_losses[-1], val_metrics[-1]))
            self.logger.info(
                '[EPOCH]: %i, [DURATION]: %.6f \n' %
                (i, time.time() - start_time))
            start_time = time.time()
        return train_losses, train_metrics, val_losses, val_metrics

    def predict(self, testloader):
        pred = []
        label = []
        stock_ID = []
        date = []
        self.best_model.eval()

        with torch.no_grad():
            for x, y, I, d in tqdm(testloader):
                if self.gpu_available:
                    x = x.float().cuda()
                    y = y.float().cuda()
                y_pred = self.best_model(x).squeeze()

                pred.extend(list(y_pred.cpu().numpy()))
                label.extend(list(y.cpu().numpy()))
                stock_ID.extend(I)
                date.extend(d)
        df_pred, df_label = self.Dense2Square(label, pred, stock_ID, date)
        IC = self.cal_IC(df_pred, df_label)
        return df_pred, df_label, IC


    def Dense2Square(self, label, pred, stockID, date):
        # print(len(label), len(pred), len(stockID), len(date))
        df_pred = pd.DataFrame(pred, columns=["pred"])
        df_pred["date"] = pd.DataFrame(date)
        df_pred["stock"] = pd.DataFrame(stockID)

        df_label = pd.DataFrame(label, columns=["label"])
        df_label["date"] = pd.DataFrame(date)
        df_label["stock"] = pd.DataFrame(stockID)

        df_pred = df_pred.pivot("stock", "date", "pred")
        df_label = df_label.pivot("stock", "date", "label")

        return df_pred, df_label


    def cal_IC(self, X, Y):
        """
        axis 0: stock
        axis 1: date
        """
        df = pd.concat([X.stack(), Y.stack()], axis=1)
        IC = df.groupby("date").apply(lambda x: x.corr("spearman").iloc[0, 1]).mean()

        return IC


if __name__ == "__main__":
    root = "../DDGAD/alpha_data/"
    alpha_daily = scio.loadmat(root + 'alpha_daily.mat')
    trade_dates = alpha_daily['dailyinfo']['dates'][0][0].squeeze()
    trade_dates = pd.to_datetime(trade_dates - 719529, unit='D')
    trade_dates = [str(i)[:5] + str(i)[5:8] + str(i)[8:10] for i in trade_dates]

    trade_dates_all = pd.to_datetime(trade_dates)
    trade_dates = trade_dates_all[trade_dates_all > pd.to_datetime("20181231", format='%Y%m%d')]

    alpha = scio.loadmat(root + 'alpha.mat')

    wind = []
    for i in alpha['basicinfo']['stock_number_wind'][0][0]:
        wind.append(i[0][0])

    alpha_daily = scio.loadmat(root + 'alpha_daily.mat')
    close_adj = alpha_daily['dailyinfo']['close_adj'][0][0].squeeze()
    close_adj = pd.DataFrame(close_adj, index=wind, columns=trade_dates_all).T

    close = alpha_daily['dailyinfo']['close'][0][0].squeeze()
    close = pd.DataFrame(close, index=wind, columns=trade_dates_all).T

    alpha_daily1 = scio.loadmat(root + 'alpha_daily_1.mat')

    Open = alpha_daily1['dailyinfo_1']['open'][0][0].squeeze()
    Open = pd.DataFrame(Open, index=wind, columns=trade_dates_all).T

    open_adj = close_adj / close * Open
    df_r = open_adj.pct_change(axis=0, periods=1).shift(-2)

    df_r = df_r.replace(np.inf, np.nan)
    df_r = df_r.apply(lambda x: (x - x.mean()) / x.std(), axis=1)


    # 训练日内模型
    class Param:
        def __init__(self):
            self.feature_num = 18
            self.model_name = 'AlphaNet_min_intraday'
            self.epochs = 200
            self.max_patience = 20
            self.lr = 0.00005
            self.gpu_id = 0
            self.objective = 0
            self.model_load_dir = "./20220824/"
            self.model_save_dir = "./20220824/"
            self.log_name = "xx11.log"
            self.batch_size = 8192

    params = Param()

    # train_path_list = sorted(path_list)[100:300]
    # valid_path_list = sorted(path_list)[305:350]
    # test_path_list = sorted(path_list)[355:]

    tasks = []
    look_back = 1
    for i in range(0, len(trade_dates) - 30, ):
        tasks.append({
            "name": trade_dates[i + look_back - 1],
            "feature": trade_dates[i:i + look_back],
            "label": trade_dates[i + look_back:i + look_back + 2],
        })
    print(len(tasks))

    data_dir = "./kline5/"
    train_tasks = tasks[425:550]
    valid_tasks = tasks[550:610]

    train_min_dataset = MinDataset(data_dir, train_tasks, params.feature_num, look_back, df_r)
    train_min_dataloader = DataLoader(train_min_dataset, 8, collate_fn=collate_fn, num_workers=4, shuffle=True, )

    valid_min_dataset = MinDataset(data_dir, valid_tasks, params.feature_num, look_back, df_r)
    valid_min_dataloader = DataLoader(valid_min_dataset, 8, collate_fn=collate_fn, num_workers=4, shuffle=False, )

    # test_min_dataloader = DataLoader(test_min_dataset, 5, collate_fn = collate_fn, num_workers=0)
    trainer = trainModels(params, )
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp_wrapper = lp(trainer.train)
    # lp.add_function(trainer.model.extractor_1.forward)
    # train_losses, train_metrics, val_losses, val_metrics = lp_wrapper(train_min_dataloader, valid_min_dataloader)
    # lp.print_stats()
    train_losses, train_metrics, val_losses, val_metrics = trainer.train(train_min_dataloader, valid_min_dataloader)


    # 训练日间模型
    class Param:
        def __init__(self):
            self.feature_num = 18
            self.model_name = 'AlphaNet_min_interday'
            self.epochs = 200
            self.max_patience = 20
            self.lr = 0.00005
            self.gpu_id = 0
            self.objective = 0
            self.model_load_dir = "./20220824/"
            self.model_save_dir = "./20220824/"
            self.log_name = "xx12.log"
            self.batch_size = 8192

    params = Param()
    tasks = []
    look_back = 5
    for i in range(0, len(trade_dates) - 30, ):
        tasks.append({
            "name": trade_dates[i + look_back - 1],
            "feature": trade_dates[i:i + look_back],
            "label": trade_dates[i + look_back:i + look_back + 2],
        })
    print(len(tasks))

    data_dir = "./kline5/"
    train_tasks = tasks[425:550]
    valid_tasks = tasks[550:610]

    train_min_dataset = MinDataset(data_dir, train_tasks, params.feature_num, look_back, df_r)
    train_min_dataloader = DataLoader(train_min_dataset, 8, collate_fn=collate_fn, num_workers=8, shuffle=True, )

    valid_min_dataset = MinDataset(data_dir, valid_tasks, params.feature_num, look_back, df_r)
    valid_min_dataloader = DataLoader(valid_min_dataset, 8, collate_fn=collate_fn, num_workers=8, shuffle=False, )

    # test_min_dataloader = DataLoader(test_min_dataset, 5, collate_fn = collate_fn, num_workers=0)
    trainer = trainModels(params, )
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp_wrapper = lp(trainer.train)
    # lp.add_function(trainer.model.extractor_1.forward)
    # train_losses, train_metrics, val_losses, val_metrics = lp_wrapper(train_min_dataloader, valid_min_dataloader)
    # lp.print_stats()
    train_losses, train_metrics, val_losses, val_metrics = trainer.train(train_min_dataloader, valid_min_dataloader)