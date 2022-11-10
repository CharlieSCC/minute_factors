# %%
from DataLoader import DataLoader, Underlying
from MiniteFactors import CloseVolPropFactor, DownwardVolaPropFactor, SkewnessKurtosisFactor, VolPriceCorrFactor, \
     AvgOutFlowPropFactor, TimeSectionPropFactor, TimeSectionStatisticFactor, TimeSectionPriceCorrFactor, \
     PriceRangeStatisticFactor, read_daily_factor, ImprovedReveralFactor
import pandas as pd
import numpy as np
import datetime as dt
import scipy.io as scio


# import ray
import os
import time
from functools import reduce, wraps
from multiprocessing import Pool, Queue, Process
import functools
from tqdm import tqdm
import pickle as pkl
# from joblib import Parallel, delayed

# %%
class Timer(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        start = time.time()
        ret = self.func(*args)
        return ret, time.time() - start

# /opt/anaconda3/bin/python /app/mount/code/intraday/calc_factor.py
def func_time(func):
    @wraps(func)
    def inner(*args, **kw):
        print('开始运行的时间为：', time.strftime('%Y:%m:%d %H:%M:%S', time.localtime()))
        start = time.time()
        result = func(*args, **kw)
        print('结束运行的时间为：', time.strftime('%Y:%m:%d %H:%M:%S', time.localtime()))
        print('运行时长为：{:6.4f}秒.'.format(time.time() - start))
        return result

    return inner

# ray.init(num_cpus=8)

# downward_vola_prop_factor = DownwardVolaPropFactor(20)
# skewness_kurtosis_factor = SkewnessKurtosisFactor(20)
# vol_price_corr_factor = VolPriceCorrFactor(20)
# avg_out_flow_prop_factor = AvgOutFlowPropFactor(20)
# factor_list = [close_vol_prop_factor, downward_vola_prop_factor, skewness_kurtosis_factor, vol_price_corr_factor, avg_out_flow_prop_factor]


# avg_out_flow_prop = AvgOutFlowPropFactor(1)
# vol_price_corr = VolPriceCorrFactor(1)
# time_section_prop_open = TimeSectionPropFactor(feature_name='TotalVolumeTrade', section_start='093000000', section_end='100000000')
# time_section_prop_close = TimeSectionPropFactor(feature_name='TotalVolumeTrade', section_start='143000000', section_end='150000000')

# time_section_return_avg_daily = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='avg', section_start='093000000', section_end='150000000', T=1, N=1)
# time_section_return_avg_open = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='avg', section_start='093000000', section_end='100000000', T=1, N=1)
# time_section_return_avg_close = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='avg', section_start='143000000', section_end='150000000', T=1, N=1)
# time_section_return_std_daily = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='std', section_start='093000000', section_end='150000000', T=1, N=1)
# time_section_return_std_open = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='std', section_start='093000000', section_end='100000000', T=1, N=1)
# time_section_return_std_close = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='std', section_start='143000000', section_end='150000000', T=1, N=1)
# time_section_return_skew_daily = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='skew', section_start='093000000', section_end='150000000', T=1, N=1)
# time_section_return_skew_open = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='skew', section_start='093000000', section_end='100000000', T=1, N=1)
# time_section_return_skew_close = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='skew', section_start='143000000', section_end='150000000', T=1, N=1)
# time_section_return_kurt_daily = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='kurt', section_start='093000000', section_end='150000000', T=1, N=1)
# time_section_return_kurt_open = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='kurt', section_start='093000000', section_end='100000000', T=1, N=1)
# time_section_return_kurt_close = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='kurt', section_start='143000000', section_end='150000000', T=1, N=1)

# external_dict = read_daily_factor(os.path.join(root, 'tquant_factors' + os.sep + 'free_float_shares.csv'))
# time_section_turnover_std_close = TimeSectionStatisticFactor(feature_name='TotalVolumeTrade', stat_method='std',
#                                                                 section_start='143000000', section_end='150000000',
#                                                                 T=1, N=0, external=external_dict, external_name='turnover')
# time_section_turnover_std_open = TimeSectionStatisticFactor(feature_name='TotalVolumeTrade', stat_method='std',
#                                                             section_start='093000000', section_end='100000000',
#                                                             T=1, N=0, external=external_dict,
#                                                             external_name='turnover')
# time_section_price_turnover_corr_daily = TimeSectionPriceCorrFactor('TotalVolumeTrade', section_start='093000000',
#                                                                     section_end='150000000',
#                                                                     T=1, external=external_dict, external_name='turnover_corr')
# price_range_turnover_std_high = PriceRangeStatisticFactor('TotalVolumeTrade', 'std', 0.5, high=True, T=1, N=0,
#                                                             external=external_dict, external_name='turnover')
# price_range_turnover_mean_high = PriceRangeStatisticFactor('TotalVolumeTrade', 'avg', 0.5, high=True, T=1, N=0,
#                                                             external=external_dict, external_name='turnover')
# factor_list = [avg_out_flow_prop, time_section_prop_open, time_section_prop_close, time_section_return_avg_daily, time_section_return_avg_open, 
#                time_section_return_avg_close, time_section_return_std_daily, time_section_return_std_open, time_section_return_std_close, 
#                time_section_return_skew_daily, time_section_return_skew_open, time_section_return_skew_close, time_section_return_kurt_daily, 
#                time_section_return_kurt_open, time_section_return_kurt_close]

# factor_list = [avg_out_flow_prop, time_section_prop_open, time_section_prop_close, 
#                time_section_return_std_daily, time_section_return_skew_daily, 
#                time_section_return_kurt_daily]

# factor_list = [vol_price_corr, time_section_return_skew_open, time_section_return_skew_close]
# factor_list = [time_section_turnover_std_close, time_section_turnover_std_open, time_section_price_turnover_corr_daily, price_range_turnover_std_high, 
#                price_range_turnover_mean_high]

# %%
# @ray.remote
# def single_stock_run(file_path: str):
#     underlying = data_loader.filter(file_path)
#     return close_vol_prop_factor.cal(underlying)


# @func_time
def multiple_factor_run(file_path: str):
    """
    :param file_path: 每个股票数据的读取路径
    :return: 一个包含该股票所有factor计算结果的字典，
    字典里每个factor的结果结构为(stock_name: str, date: np.ndarray, result: np.ndarray)
    """
    underlying = data_loader.filter(file_path)
    factor_result_dict = {}
    
    for factor in factor_list:
        factor_result_dict[factor.factor_name] = factor.cal(underlying)
    # def single_factor(temp_factor):
    #     factor_result_dict[temp_factor.factor_name] = temp_factor.cal(underlying)
    # Parallel(n_jobs=6)(delayed(single_factor)(factor) for factor in factor_list)
    print(list(factor_result_dict.keys()))
    return factor_result_dict


def multiple_factor_run_queue(shared_queue: Queue):
    """
    :param shared_queue: 共享的 Queue 队列
    :return: 一个包含该股票所有factor计算结果的字典，
    字典里每个factor的结果结构为(stock_name: str, date: np.ndarray, result: np.ndarray)
    """
    underlying = shared_queue.get()
    if underlying != -1:
        stock_name = underlying.name
        factor_result_dict = {}
        for factor in factor_list:
            factor_result_dict[factor.factor_name] = factor.cal(underlying)
        return stock_name, factor_result_dict
    else:
        return 'Finished', -1


def multiple_stock_run(file_path_list: list):
    """
    计算在file_path_list的股票的因子结果并按照 股票名字：值 的键值对保存结果到字典
    :param file_path_list: 股票数据的读取路径列表
    :return: 一个字典，包含多个股票的计算结果
    """
    stock_dict = {}
    for file_path in file_path_list:
        stock_name = '.'.join(file_path.split(os.sep)[-1].split('.')[:2])
        stock_dict[stock_name] = multiple_factor_run(file_path)
    return stock_dict



@func_time
def results_to_df(results: list) -> pd.DataFrame:
    df_list = []
    for result in results:
        df_list.append(pd.DataFrame({'tick': result[1], result[0]: result[2]}))
    df_result = reduce(lambda left, right: pd.merge(left, right, on='tick', how='outer'), df_list)
    return df_result

@func_time
def combine_to_df(results: list) -> pd.DataFrame:
    df_lt = []
    for result in tqdm(results):
        if result is not None:
            print(result[0])
            try:
                temp = pd.DataFrame(result[2], columns=[result[0]], index=result[1])
            except:
                temp = pd.DataFrame([result[2]], columns=[result[0]], index=result[1])
            if not temp.isna().all()[0]:
                df_lt.append(temp)
    return pd.concat(df_lt, axis=1).T


def integrate_to_factor_dict(stock_factor_dict: dict) -> dict:
    factor_dict = {}
    for stock_dict in stock_factor_dict.values():
        for factor_name, factor in stock_dict.items():
            if factor_dict.get(factor_name, -1) == -1:
                factor_dict[factor_name] = []
                factor_dict[factor_name].append(factor)
            else:
                factor_dict[factor_name].append(factor)
    return factor_dict


# %%
# @func_time
# def main():
#     results = ray.get([single_stock_run.remote(path) for path in data_loader.path_list[:24]])
#     # close_vol_prop_factor_df = results_to_df(results)
#     close_vol_prop_factor_df = combine_to_df(results)
#     close_vol_prop_factor_df.to_csv('close_vol_prop_factor.csv')
#     # print(close_vol_prop_factor_df.info())
#     # print(close_vol_prop_factor_df.describe())
#     # print(close_vol_prop_factor_df)
#     # print(results[0:5])
#     # ray.get([ray.remote(singal_save_to_pickle).remote(c, save_path) for c in codes])


def single_run(exist_stock_dict: dict = None):
    if exist_stock_dict is None:
        stock_dict = {}
        pbar = tqdm(data_loader.path_list)
    else:
        stock_dict = exist_stock_dict
        exist_codes = list(stock_dict.keys())
        print(f'all codes length: {len(data_loader.path_list)}')
        sub_path_list = [path for path in data_loader.path_list if path.split(os.sep)[-1][:-4] not in exist_codes]
        print(f"after subtraction, we have codes length: {len(sub_path_list)}")
        pbar = tqdm(sub_path_list)
    count = 0
    for path in pbar:      # [:24]:
        stock_name = '.'.join(path.split(os.sep)[-1].split('.')[:2])
        pbar.set_description(f'Processing stock: {stock_name}')
        stock_dict[stock_name] = multiple_factor_run(path)
        if count % 100 == 0:
            print(list(stock_dict.keys())[-1], f'is saved and the index number is {count}')
            with open(os.path.join(root, 'temp_save.pkl'), 'wb') as f:
                pkl.dump(stock_dict, f)
        count += 1
    return stock_dict


def resource_allocate_run(cpu_numbs: int):
    results = []
    p = Pool(cpu_numbs)
    path_list = data_loader.path_list       # [:24]
    stock_num = len(path_list)
    par_list = [path_list[x::cpu_numbs] for x in range(stock_num)]
    for i in range(cpu_numbs):
        results.append(p.apply_async(multiple_stock_run, args=(par_list[i],)))
    p.close()
    p.join()
    out_list = []
    for i in results:
        out_list.append(i.get())
    # print(out_list)
    return out_list


def main(results, out_path: str):
    if isinstance(results, dict):
        factor_dict = integrate_to_factor_dict(results)
        for factor_name, factor_list in factor_dict.items():
            df = combine_to_df(factor_list)
            df = transform(df)
            df.to_csv(os.path.join(out_path, factor_name + '.csv'), index=False)
    elif isinstance(results, list):
        stock_dict = dict()
        for dict_ in results:
            # print(dict_)
            stock_dict.update(dict_)
        factor_dict = integrate_to_factor_dict(stock_dict)
        for factor_name, factor_list in factor_dict.items():
            df = combine_to_df(factor_list)
            df = transform(df)
            df.to_csv(os.path.join(out_path, factor_name + '.csv'))
    else:
        print('Wrong input type of results!')

def transform(df):
    df.columns = pd.to_datetime(df.columns.astype(str))
    alpha = scio.loadmat('/app/mount/code/intraday/alpha.mat')
    lt = []
    for i in alpha["basicinfo"]["stock_number_wind"][0][0]:
        lt.append(i[0][0])
    temp = pd.DataFrame([lt,list(range(1,1+len(lt)))]).T
    temp.columns =["code", "num"]
    dic = temp.set_index('code').to_dict()["num"]
    df = df[df.index.isin(lt)] # alpha.mat 新股未更新
    df.index = df.reset_index()["index"].apply(lambda x: dic[x])
    template = pd.DataFrame(index=list(range(1,len(lt)+1)))
    df = pd.concat([template,df.sort_index()],axis=1)
    return df

def track_exist_result():
    with open(os.path.join(root, 'temp_save.pkl'), 'rb') as f:
        stock_dict = pkl.load(f)
        print('exist data loaded!')
    return stock_dict


# %%
if __name__ == '__main__':
    start = time.time()
    root = '/app/mount/code/intraday/'

    used_col_list = ['HTSCSecurityID', 'MDDate', 'MDTime', 'OpenPx',
                    'ClosePx', 'HighPx', 'LowPx', 'NumTrades',
                    'TotalVolumeTrade', 'TotalValueTrade']
    # data_loader = DataLoader(root, '20220701 092500000', '20220815 150000250', used_col_list)
    data_loader = DataLoader(root, '20130101 092500000', '20220822 150000250', used_col_list)
    close_vol_prop_factor = CloseVolPropFactor('143000000', '150000000', 20)
    improved_reveral_factor = ImprovedReveralFactor('092500000', '100000000', '150000000', 20)
    vol_price_corr_factor = VolPriceCorrFactor(20)
    skewness_kurtosis_factor = SkewnessKurtosisFactor(20)

    downward_vol_prop_factor = DownwardVolaPropFactor(20)
    avg_out_flow_prop_factor = AvgOutFlowPropFactor(20)
    
    

    # avg_out_flow_prop = AvgOutFlowPropFactor(1)
    # vol_price_corr = VolPriceCorrFactor(1)
    # time_section_prop_open = TimeSectionPropFactor(feature_name='TotalVolumeTrade', section_start='093000000', section_end='100000000')
    # time_section_prop_close = TimeSectionPropFactor(feature_name='TotalVolumeTrade', section_start='143000000', section_end='150000000')

    # time_section_return_avg_daily = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='avg', section_start='093000000', section_end='150000000', T=1, N=1)
    # time_section_return_avg_open = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='avg', section_start='093000000', section_end='100000000', T=1, N=1)
    # time_section_return_avg_close = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='avg', section_start='143000000', section_end='150000000', T=1, N=1)
    # time_section_return_std_daily = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='std', section_start='093000000', section_end='150000000', T=1, N=1)
    # time_section_return_std_open = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='std', section_start='093000000', section_end='100000000', T=1, N=1)
    # time_section_return_std_close = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='std', section_start='143000000', section_end='150000000', T=1, N=1)
    # time_section_return_skew_daily = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='skew', section_start='093000000', section_end='150000000', T=1, N=1)
    # time_section_return_skew_open = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='skew', section_start='093000000', section_end='100000000', T=1, N=1)
    # time_section_return_skew_close = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='skew', section_start='143000000', section_end='150000000', T=1, N=1)
    # time_section_return_kurt_daily = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='kurt', section_start='093000000', section_end='150000000', T=1, N=1)
    # time_section_return_kurt_open = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='kurt', section_start='093000000', section_end='100000000', T=1, N=1)
    # time_section_return_kurt_close = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='kurt', section_start='143000000', section_end='150000000', T=1, N=1)

    # factor_list = [ downward_vola_prop_factor,
    #                 avg_out_flow_prop_factor,
    #                 avg_out_flow_prop,
    #                 vol_price_corr,
    #                 time_section_prop_open, 
    #                 time_section_prop_close,
    #                 time_section_return_avg_daily,
    #                 time_section_return_avg_open,
    #                 time_section_return_avg_close,
    #                 time_section_return_std_daily,
    #                 time_section_return_std_open,
    #                 time_section_return_std_close,
    #                 time_section_return_skew_daily,
    #                 time_section_return_skew_open,
    #                 time_section_return_skew_close,
    #                 time_section_return_kurt_daily,
    #                 time_section_return_kurt_open,
    #                 time_section_return_kurt_close
    # ]
    # external_dict = read_daily_factor(os.path.join(root, 'tquant_factors' + os.sep + 'free_float_shares.csv'))
    # time_section_turnover_std_close = TimeSectionStatisticFactor(feature_name='TotalVolumeTrade', stat_method='std',
    #                                                                 section_start='143000000', section_end='150000000',
    #                                                                 T=1, N=0, external=external_dict, external_name='turnover')
    # time_section_turnover_std_open = TimeSectionStatisticFactor(feature_name='TotalVolumeTrade', stat_method='std',
    #                                                             section_start='093000000', section_end='100000000',
    #                                                             T=1, N=0, external=external_dict,
    #                                                             external_name='turnover')
    # time_section_price_turnover_corr_daily = TimeSectionPriceCorrFactor('TotalVolumeTrade', section_start='093000000',
    #                                                                     section_end='150000000',
    #                                                                     T=1, external=external_dict, external_name='turnover_corr')
    # price_range_turnover_std_high = PriceRangeStatisticFactor('TotalVolumeTrade', 'std', 0.5, high=True, T=1, N=0,
    #                                                             external=external_dict, external_name='turnover')
    # price_range_turnover_mean_high = PriceRangeStatisticFactor('TotalVolumeTrade', 'avg', 0.5, high=True, T=1, N=0,
    #                                                             external=external_dict, external_name='turnover')
    # factor_list = [avg_out_flow_prop, time_section_prop_open, time_section_prop_close, time_section_return_avg_daily, time_section_return_avg_open, 
    #                time_section_return_avg_close, time_section_return_std_daily, time_section_return_std_open, time_section_return_std_close, 
    #                time_section_return_skew_daily, time_section_return_skew_open, time_section_return_skew_close, time_section_return_kurt_daily, 
    #                time_section_return_kurt_open, time_section_return_kurt_close]

    # factor_list = [avg_out_flow_prop, time_section_prop_open, time_section_prop_close, 
    #                time_section_return_std_daily, time_section_return_skew_daily, 
    #                time_section_return_kurt_daily]

    # factor_list = [vol_price_corr, time_section_return_skew_open, time_section_return_skew_close]
    # factor_list = [time_section_turnover_std_close, time_section_turnover_std_open, time_section_price_turnover_corr_daily, price_range_turnover_std_high, 
    #                price_range_turnover_mean_high]




    factor_list = [close_vol_prop_factor, improved_reveral_factor, vol_price_corr_factor, skewness_kurtosis_factor,
    downward_vola_prop_factor, avg_out_flow_prop_factor]
    # result = resource_allocate_run(8)
    # exist_stock_dict = track_exist_result()
    result = single_run(None)

    # result = result[]
    out_path = os.path.join(root, 'factors2')
    main(result, out_path)
    print('总运行时长为：{:6.4f}秒'.format(time.time() - start))
    # 单进程前24个股票的单因子 close_vol_prop_factor 总运行时长为：51.6675秒,  70.9586秒
    # 单进程前24个股票的4个因子 close_vol_prop_factor, downward_vola_prop_factor, skewness_kurtosis_factor, vol_price_corr_factor 总运行时长为：118.1874秒,  82.4067秒
    # 多进程前24个股票的单因子 close_vol_prop_factor 总运行时长为：97.0651秒  100.1654秒
    # 多进程前24个股票的4个因子 close_vol_prop_factor, downward_vola_prop_factor, skewness_kurtosis_factor, vol_price_corr_factor 总运行时长为：132.1826秒 138.4714秒
    # joblib 测试 n_jobs = 4 6因子 6个股票的时长  137.6009秒 107.1337秒
    # joblib 测试 n_jobs = 6 6因子 6个股票的时长  111.8511秒
    # joblib 测试 n_jobs = 8 6因子 6个股票的时长  113.9147秒
    # joblib 测试 原始       6因子 6个股票的时长  108.8785秒