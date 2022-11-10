# %%
import pandas as pd
import numpy as np
import os
import glob
import sys
import time
import functools
import threading
from multiprocessing import Manager
from tqdm import tqdm


def func_time(func):
    @functools.wraps(func)
    def inner(*args, **kw):
        # print('数据加载开始运行的时间为：', time.strftime('%Y:%m:%d %H:%M:%S', time.localtime()))
        start = time.time()
        result = func(*args, **kw)
        # print('数据加载结束运行的时间为：', time.strftime('%Y:%m:%d %H:%M:%S', time.localtime()))
        print(f'数据加载运行时长为：{time.time() - start}秒.')
        return result

    return inner

# %%
class Underlying:
    def __init__(self, name: str, data: pd.DataFrame):
        self.name = name
        self.data = data


class DataLoader:

    def __init__(self, root_path: str, start: str, end: str, col_list: list, dir_name='kline3'):
        self.start = pd.to_datetime(start, format='%Y%m%d %H%M%S%f')
        self.end = pd.to_datetime(end, format='%Y%m%d %H%M%S%f')
        self.col_list = col_list
        self.path_list = DataLoader.get_file_names(root_path, dir_name, '.S', 'pkl')
        self.underlying_list = []
        self.underlying_queue = Manager().Queue(20)

    @staticmethod
    def get_file_names(root_path: str, dir_name: str, file_name: str, file_type: str) -> list:
        if os.path.isdir(root_path):
            file_list = glob.glob(os.path.join(root_path, dir_name) + '/*' + file_name + '*' + '.' + file_type)
            return file_list
        else:
            raise Exception("wrong paths are inputted!!! Please reconstruct a new root path.")

    @func_time
    def filter(self, file_path: str):
        temp_df = pd.read_pickle(file_path)
        temp_df = temp_df[self.col_list]
        try:
            stock_name = file_path[-13:-4]# temp_df['HTSCSecurityID'][0].values[0]
        except:
            #print(temp_df['HTSCSecurityID'])
            stock_name = temp_df['HTSCSecurityID'][0]
        temp_df['m_tick'] = pd.to_datetime(temp_df.MDDate + ' ' + temp_df.MDTime, format='%Y%m%d %H%M%S%f')
        temp_df = temp_df.loc[(temp_df['m_tick'] >= self.start) & (temp_df['m_tick'] <= self.end), :]
        temp_df['MDDate'] = pd.to_numeric(temp_df['MDDate'])
        temp_df['MDTime'] = pd.to_numeric(temp_df['MDTime'])
        temp_df.drop(['HTSCSecurityID'], axis=1, inplace=True)
        print(f"Loading {file_path.split(os.sep)[-1]}'s data'")
        return Underlying(stock_name, temp_df.copy())

    def multi_filter(self, file_path_list: list):
        underlying_list = []
        for file_path in file_path_list:
            underlying_list.append(self.filter(file_path))
        self.underlying_list.extend(underlying_list)
        # print(underlying_list)

    def multi_filter_queue(self, file_path_list: list):
        for file_path in file_path_list:
            self.underlying_queue.put(self.filter(file_path))
        self.underlying_queue.put(-1)

    def get_all(self, threads_num: int = 3):
        threads = []
        par_list = [self.path_list[x::threads_num] for x in range(len(self.path_list))]
        for t in range(0, threads_num):
            thread = threading.Thread(target=self.multi_filter, args=(par_list[t],))
            thread.start()
            threads.append(thread)
        for thr in threads:
            # thr.start()
            thr.join()

    def get_all_queue(self, threads_num: int = 3):
        threads = []
        par_list = [self.path_list[x::threads_num] for x in range(len(self.path_list))]
        for t in range(0, threads_num):
            thread = threading.Thread(target=self.multi_filter_queue, args=(par_list[t],))
            thread.start()
            threads.append(thread)
        for thr in threads:
            # thr.start()
            thr.join()

    def get_all_queue_serial(self):
        pbar = tqdm(self.path_list)
        for path in pbar:
            stock_name = '.'.join(path.split(os.sep)[-1].split('.')[:2])
            pbar.set_description(f'Processing stock: {stock_name}')
            self.underlying_queue.put(self.filter(path))
        while True:
            if self.underlying_queue.qsize() < 20:
                self.underlying_queue.put(-1)
                time.sleep(0.1)
            else:
                print("Task finished!")
                break


# %%
if __name__ == '__main__':
    root = 'intraday'
# %%
    used_col_list = ['HTSCSecurityID', 'MDDate', 'MDTime', 'OpenPx',
                     'ClosePx', 'HighPx', 'LowPx', 'NumTrades',
                     'TotalVolumeTrade', 'TotalValueTrade']
    data_loader = DataLoader(root, '20170101 092500000', '20211231 150000000', used_col_list)
    print(data_loader.path_list)

# %%
    df = pd.read_pickle(os.path.join(root, 'kline3' + os.sep + '000001.SZ.pkl'))
    start = time.time()
    data1 = data_loader.filter(data_loader.path_list[0])
    data2 = data_loader.filter(data_loader.path_list[1])
    data3 = data_loader.filter(data_loader.path_list[1])
    print(f'串行耗时{time.time() - start} 秒')


# %%
    start = time.time()
    # data_loader.get_all_queue(3)
    data_loader.get_all_queue_serial()
    for i in range(3):
        data = data_loader.underlying_queue.get()
        if data != -1:
            data = data.data
            print(data)
        # print(data[0], data[1], data[2])
    print(f'多线程总耗时为{time.time() - start}')