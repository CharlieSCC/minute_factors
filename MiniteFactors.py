# %%
from DataLoader import DataLoader, Underlying
import pandas as pd
import numpy as np
import datetime as dt
from abc import ABC, abstractmethod
import time
import functools
import os


# %%
def func_time(func):
    @functools.wraps(func)
    def inner(*args, **kw):
        # print('因子开始运行的时间为：', time.strftime('%Y:%m:%d %H:%M:%S', time.localtime()))
        start = time.time()
        try:
            result = func(*args, **kw)
        except:
            result = None 
        # print('因子结束运行的时间为：', time.strftime('%Y:%m:%d %H:%M:%S', time.localtime()))
        print('运行时长为：{:6.4f}秒.'.format(time.time() - start))
        return result

    return inner


class Factor(ABC):
    @abstractmethod
    def cal(self, underlying: Underlying) -> tuple:
        pass

class ImprovedReveralFactor(Factor):
    """
    改进反转因子
    """
    def __init__(self, section_start: str, section_medium:str, section_end: str, T: int):
        self.section_start = dt.datetime.strptime(section_start, '%H%M%S%f').time()
        self.section_medium = dt.datetime.strptime(section_medium,  '%H%M%S%f').time()
        self.section_end = dt.datetime.strptime(section_end, '%H%M%S%f').time()
        self.factor_name = 'ImprovedReveralFactor'
        self.T = T
        
    def __str__(self, ):
        return 'ImprovedReveralFactor'

    @func_time
    def cal(self, underlying: Underlying):
        stock_name = underlying.name
        temp_df = underlying.data[['MDDate', 'm_tick', 'OpenPx', 'ClosePx']].copy()
        # temp_df_group = temp_df.groupby(['MDDate'])
        temp_df_start = temp_df.loc[temp_df['m_tick'].dt.time == self.section_start,:]
        temp_df_medium = temp_df.loc[temp_df['m_tick'].dt.time == self.section_medium,:]
        temp_df_end = temp_df.loc[temp_df['m_tick'].dt.time == self.section_end,:] 
        if len(temp_df_end) != len(temp_df_start) or len(temp_df_end) != len(temp_df_medium):
            # 有的时候早盘或者尾盘没有数据
            del temp_df_start, temp_df_medium, temp_df_end
            return stock_name, temp_df['MDDate'].values, np.full(len(temp_df['MDDate'].values), np.nan)
        JumpFactor = np.log(np.abs(temp_df_start["OpenPx"].values/temp_df_end["ClosePx"].shift(1).values))
        IntraDayFactor = np.log(temp_df_end["ClosePx"].values/temp_df_medium["OpenPx"].values)
        StructralFactor = JumpFactor * 0.4 + 0.6 * IntraDayFactor
        result = pd.Series(StructralFactor, index=temp_df_start['MDDate'])
        if self.T != 1:
            result = result.rolling(self.T, min_periods=1).mean()
            result = result.iloc[self.T - 1:]
        del temp_df, temp_df_start, temp_df_medium, temp_df_end, JumpFactor, IntraDayFactor
        return stock_name, result.index.values, result.values.squeeze()

class CloseVolPropFactor(Factor):
    """
    尾盘成交占比
    """

    def __init__(self, section_start: str, section_end: str, T: int):
        self.section_start = dt.datetime.strptime(section_start, '%H%M%S%f').time()
        self.section_end = dt.datetime.strptime(section_end, '%H%M%S%f').time()
        self.T = T
        self.factor_name = 'CloseVolPropFactor'
        # self.value_list = []

    def __str__(self):
        return 'CloseVolPropFactor'

    @func_time
    def cal(self, underlying: Underlying):
        stock_name = underlying.name
        temp_df = underlying.data[['MDDate', 'm_tick', 'TotalVolumeTrade']].copy()
        total_volume = temp_df.groupby(['MDDate'])['TotalVolumeTrade'].sum()  # sum()  apply(np.nansum)
        total_dates = total_volume.index
        temp_df = temp_df.loc[
                  (temp_df['m_tick'].dt.time >= self.section_start) & (temp_df['m_tick'].dt.time <= self.section_end),
                  :]
        sub_volume = temp_df.groupby(['MDDate'])['TotalVolumeTrade'].sum()  # sum()  apply(np.nansum)
        sub_volume = sub_volume.reindex(total_dates)
        del temp_df
        result = pd.Series(sub_volume.values / total_volume.values, index=sub_volume.index)
        del sub_volume
        del total_volume
        result = result.rolling(self.T, min_periods=1).mean()
        result = result.iloc[self.T - 1:]
        print(stock_name, self, 'finished!')
        return stock_name, result.index.values, result.values



class TimeSectionPropFactor(Factor):
    """
    给定日间某单个变量的名称 可计算其选定时间段日度占比
    同时可以以选定是否需要做日度平滑  （给定T为大于1的值即表示做T日平滑，默认T=1，不做平滑）
    时间范围默认是早盘：section_start='093000000', section_end='100000000'
    """

    def __init__(self, feature_name: str, section_start: str = '093000000',
                 section_end: str = '100000000', T: int = 1):
        self.section_start = dt.datetime.strptime(section_start, '%H%M%S%f').time()
        self.section_end = dt.datetime.strptime(section_end, '%H%M%S%f').time()
        self.feature_name = feature_name
        self.T = T
        self.factor_name = f'TimeSectionPropFactor_{self.section_start.hour}{self.section_start.minute}-{self.section_end.hour}{self.section_end.minute}_{self.feature_name}'

    def __str__(self):
        return self.factor_name

    @func_time
    def cal(self, underlying: Underlying):
        stock_name = underlying.name
        temp_df = underlying.data[['MDDate', 'm_tick', self.feature_name]].copy()
        total_value = temp_df.groupby(['MDDate'])[self.feature_name].sum()
        total_dates = total_value.index
        temp_df = temp_df.loc[
                  (temp_df['m_tick'].dt.time >= self.section_start) & (temp_df['m_tick'].dt.time <= self.section_end),
                  :]
        sub_value = temp_df.groupby(['MDDate'])[self.feature_name].sum()
        sub_value = sub_value.reindex(total_dates)
        result = pd.Series(sub_value.values / total_value.values, index=sub_value.index)
        if self.T != 1:
            result = result.rolling(self.T, min_periods=1).mean()
            result = result.iloc[self.T - 1:]

        print(stock_name, self, 'finished!')
        return stock_name, result.index.values, result.values


class TimeSectionStatisticFactor(Factor):
    """
    给定日间某单个变量的名称 可选择性计算其 均值avg\偏度skew\峰度kurt\标准差std
    同时可以以选定是否需要做日度平滑  （给定T为大于1的值即表示做T日平滑，默认T=1，不做平滑）
    以及选择是否用此变量计算N分钟增长率（收益率）  （默认为0，不计算增长率，可选择N的大小做N分钟增长率计算）
    时间范围默认是日度：section_start='093000000', section_end='150000000'
    以及可以选择是否有外来数据进行辅助计算，目前仅支持选定的feature和外来数据做除法运算, external 为外来变量的插入路径，读入后会分割为key为股票
    名称，value为对应相关天数dataframe的dict
    """

    def __init__(self, feature_name: str, stat_method: str, section_start: str = '093000000',
                 section_end: str = '150000000', T: int = 1, N: int = 0, external=None, external_name=None):
        self.section_start = dt.datetime.strptime(section_start, '%H%M%S%f').time()
        self.section_end = dt.datetime.strptime(section_end, '%H%M%S%f').time()
        self.feature_name = feature_name
        self.stat_method = stat_method
        self.T = T
        self.N = N
        if external is not None and external_name is not None:
            self.external_dict = external   #read_daily_factor(external)
            self.external_name = external_name
        else:
            self.external_dict = None
        if N == 0:
            if external is None:
                self.factor_name = f'TimeSectionStatisticFactor_{self.section_start.hour}{self.section_start.minute}-{self.section_end.hour}{self.section_end.minute}_{self.stat_method}_{self.feature_name}'
            else:
                self.factor_name = f'TimeSectionStatisticFactor_{self.section_start.hour}{self.section_start.minute}-{self.section_end.hour}{self.section_end.minute}_{self.stat_method}_{self.external_name}'
        else:
            self.factor_name = f'TimeSectionStatisticFactor_{self.section_start.hour}{self.section_start.minute}-{self.section_end.hour}{self.section_end.minute}_{self.stat_method}_return'

    def __str__(self):
        return self.factor_name

    @func_time
    def cal(self, underlying: Underlying):
        stock_name = underlying.name
        temp_df = underlying.data[['MDDate', 'm_tick', self.feature_name]].copy()
        if self.external_dict is not None:
            temp_df = temp_df.merge(self.external_dict[stock_name], how='left', left_on='MDDate', right_on='mddate')
            new_col_name = self.external_dict[stock_name].columns.to_list()[-1]
            temp_df[self.feature_name] = temp_df[self.feature_name] / temp_df[new_col_name]
            temp_df.drop(['mddate', new_col_name], inplace=True, axis=1)
        if self.N != 0:
            back_sr = temp_df[self.feature_name].shift(self.N)
            temp_df[self.feature_name] = (temp_df[self.feature_name] - back_sr) / back_sr
            # temp_df.replace(np.nan, 0, inplace=True)
        total_dates = temp_df.groupby(['MDDate'])[self.feature_name].sum().index
        temp_df = temp_df.loc[
                  (temp_df['m_tick'].dt.time >= self.section_start) & (temp_df['m_tick'].dt.time <= self.section_end),
                  :]
        if self.stat_method == 'avg':
            result = temp_df.groupby(['MDDate'])[self.feature_name].mean()
        elif self.stat_method == 'std':
            result = temp_df.groupby(['MDDate'])[self.feature_name].std()
        elif self.stat_method == 'skew':
            result = temp_df.groupby(['MDDate'])[self.feature_name].skew(skipna=True)
        elif self.stat_method == 'kurt':
            result = temp_df.groupby(['MDDate'])[self.feature_name].apply(pd.Series.kurt,
                                                                          skipna=True)  # kurtosis(skipna=True)
        result = result.reindex(total_dates)
        if self.T != 1:
            result = result.rolling(self.T, min_periods=1).mean()
            result = result.iloc[self.T - 1:]

        print(stock_name, self, 'finished!')
        return stock_name, result.index.values, result.values


class TimeSectionPriceCorrFactor(Factor):

    def __init__(self, feature_name: str, section_start: str = '093000000',
                 section_end: str = '100000000', T: int = 1, external=None, external_name=None):
        self.section_start = dt.datetime.strptime(section_start, '%H%M%S%f').time()
        self.section_end = dt.datetime.strptime(section_end, '%H%M%S%f').time()
        self.feature_name = feature_name
        self.T = T
        if external is not None and external_name is not None:
            self.external_dict = external   #read_daily_factor(external)
            self.external_name = external_name
        else:
            self.external_dict = None
        if external is None:
            self.factor_name = f'TimeSectionPriceCorrFactor_{self.section_start.hour}{self.section_start.minute}-{self.section_end.hour}{self.section_end.minute}_{self.feature_name}'
        else:
            self.factor_name = f'TimeSectionPriceCorrFactor_{self.section_start.hour}{self.section_start.minute}-{self.section_end.hour}{self.section_end.minute}_{self.external_name}'

    def __str__(self):
        return self.factor_name

    @func_time
    def cal(self, underlying: Underlying):
        stock_name = underlying.name
        temp_df = underlying.data[['MDDate', 'm_tick', 'ClosePx', self.feature_name]].copy()
        new_col_name = self.external_dict[stock_name].columns.to_list()[-1]
        temp_df = temp_df.merge(self.external_dict[stock_name], how='left', left_on='MDDate', right_on='mddate')
        temp_df[self.feature_name] = temp_df[self.feature_name] / temp_df[new_col_name]
        temp_df.drop(['mddate', new_col_name], inplace=True, axis=1)
        total_dates = temp_df.groupby(['MDDate'])[self.feature_name].sum().index
        temp_df = temp_df.loc[
                  (temp_df['m_tick'].dt.time >= self.section_start) & (temp_df['m_tick'].dt.time <= self.section_end),
                  :]
        result = temp_df.groupby('MDDate')[['ClosePx', self.feature_name]].corr().unstack().iloc[:, 1]
        result = result.reindex(total_dates)
        if self.T != 1:
            result = result.rolling(self.T, min_periods=1).mean()
            result = result.iloc[self.T - 1:]
        print(stock_name, self, 'finished!')
        return stock_name, result.index.values, result.values


class PriceRangeStatisticFactor(Factor):
    """
    给定日间某单个变量的名称 可选择性计算其 均值avg\偏度skew\峰度kurt\标准差std
    同时可以以选定是否需要做日度平滑  （给定T为大于1的值即表示做T日平滑，默认T=1，不做平滑）
    以及选择是否用此变量计算N分钟增长率（收益率）  （默认为0，不计算增长率，可选择N的大小做N分钟增长率计算）
    quantile: 给定一个分位数，它会成为高低区间的划分点
    以及可以选择是否有外来数据进行辅助计算，目前仅支持选定的feature和外来数据做除法运算
    """

    def __init__(self, feature_name: str, stat_method: str, quantile: float, high: bool = True, T: int = 1, N: int = 0,
                 external=None, external_name=None):
        self.quantile = quantile
        self.feature_name = feature_name
        self.stat_method = stat_method
        self.high = high
        if high:
            self.range = 'high'
        else:
            self.range = 'low'
        self.T = T
        self.N = N
        if external is not None and external_name is not None:
            self.external_dict = external   #read_daily_factor(external)
            self.external_name = external_name
        else:
            self.external_dict = None
        if N == 0:
            if external is None:
                self.factor_name = f'PriceRangeStatisticFactor_{self.quantile}_{self.range}_{self.stat_method}_{self.feature_name}'
            else:
                self.factor_name = f'PriceRangeStatisticFactor_{self.quantile}_{self.range}_{self.stat_method}_{self.external_name}'
        else:
            self.factor_name = f'PriceRangeStatisticFactor_{self.quantile}_{self.range}_{self.stat_method}_return'

    def __str__(self):
        return self.factor_name

    @func_time
    def cal(self, underlying: Underlying):
        stock_name = underlying.name
        temp_df = underlying.data[['MDDate', 'ClosePx', self.feature_name]].copy()
        if self.external_dict is not None:
            temp_df = temp_df.merge(self.external_dict[stock_name], how='left', left_on='MDDate', right_on='mddate')
            new_col_name = self.external_dict[stock_name].columns.to_list()[-1]
            temp_df[self.feature_name] = temp_df[self.feature_name] / temp_df[new_col_name]
            temp_df.drop(['mddate', new_col_name], inplace=True, axis=1)
        if self.N != 0:
            back_sr = temp_df[self.feature_name].shift(self.N)
            temp_df[self.feature_name] = (temp_df[self.feature_name] - back_sr) / back_sr
        total_dates = temp_df.groupby(['MDDate'])[self.feature_name].sum().index
        if self.high:
            temp_df = temp_df.groupby(['MDDate']).apply(lambda x: x.loc[x['ClosePx'] >= x['ClosePx'].quantile(self.quantile), :]).reset_index(drop=True)
        else:
            temp_df = temp_df.groupby(['MDDate']).apply(lambda x: x.loc[x['ClosePx'] <= x['ClosePx'].quantile(self.quantile), :]).reset_index(drop=True)
        if self.stat_method == 'avg':
            result = temp_df.groupby(['MDDate'])[self.feature_name].mean()
        elif self.stat_method == 'std':
            result = temp_df.groupby(['MDDate'])[self.feature_name].std()
        elif self.stat_method == 'skew':
            result = temp_df.groupby(['MDDate'])[self.feature_name].skew(skipna=True)
        elif self.stat_method == 'kurt':
            result = temp_df.groupby(['MDDate'])[self.feature_name].apply(pd.Series.kurt,
                                                                          skipna=True)  # kurtosis(skipna=True)
        result = result.reindex(total_dates)
        if self.T != 1:
            result = result.rolling(self.T, min_periods=1).mean()
            result = result.iloc[self.T - 1:]

        print(stock_name, self, 'finished!')
        return stock_name, result.index.values, result.values




class SkewnessKurtosisFactor(Factor):
    """
    高频偏度
    """

    def __init__(self, T: int):
        self.T = T
        self.factor_name = 'SkewnessKurtosisFactor'

    def __str__(self):
        return 'SkewnessKurtosisFactor'

    @func_time
    def cal(self, underlying: Underlying):
        stock_name = underlying.name
        temp_df = underlying.data[['MDDate', 'ClosePx']].copy()
        back_sr = temp_df['ClosePx'].shift(1)
        temp_df['return'] = (temp_df['ClosePx'] - back_sr) / back_sr
        temp_df.replace(np.nan, 0, inplace=True)
        temp_df['cube'] = temp_df['return'] ** 3
        temp_df['square'] = temp_df['return'] ** 2
        date_len_df = temp_df.groupby('MDDate').size().copy()
        temp_df = temp_df.groupby('MDDate')[['square', 'cube']].sum()  # sum()  apply(np.nansum)
        result = np.sqrt(date_len_df) * temp_df['cube'] / temp_df['square'] ** 1.5
        result = result.rolling(self.T, min_periods=1).mean()
        result = result.iloc[self.T - 1:]
        print(stock_name, self, 'finished!')
        return stock_name, result.index.values, result.values


class DownwardVolaPropFactor(Factor):
    """
    下行波动占比
    """

    def __init__(self, T: int):
        self.T = T
        self.factor_name = 'DownwardVolaPropFactor'

    def __str__(self):
        return 'DownwardVolaPropFactor'

    @func_time
    def cal(self, underlying: Underlying):
        stock_name = underlying.name
        temp_df = underlying.data[['MDDate', 'ClosePx']].copy()
        back_sr = temp_df['ClosePx'].shift(1)
        temp_df['return'] = (temp_df['ClosePx'] - back_sr) / back_sr
        temp_df.replace(np.nan, 0, inplace=True)
        temp_df['square'] = temp_df['return'] ** 2
        temp_df['i_square'] = temp_df['square'] * vector_sign(temp_df['return'])
        date_len_df = temp_df.groupby('MDDate').size().copy()
        temp_df = temp_df.groupby('MDDate')[['square', 'i_square']].sum()
        result = np.sqrt(date_len_df) * temp_df['i_square'] / temp_df['square']
        result = result.rolling(self.T, min_periods=1).mean()
        result = result.iloc[self.T - 1:]
        print(stock_name, self, 'finished!')
        return stock_name, result.index.values, result.values


class VolPriceCorrFactor(Factor):
    """
    量价相关性
    """

    def __init__(self, T: int):
        self.T = T
        self.factor_name = 'VolPriceCorrFactor'

    def __str__(self):
        return 'VolPriceCorrFactor'

    @func_time
    def cal(self, underlying: Underlying):
        stock_name = underlying.name
        temp_df = underlying.data[['MDDate', 'ClosePx', 'TotalVolumeTrade']].copy()
        temp_df['TotalVolumeTrade'] = temp_df['TotalVolumeTrade'] / temp_df.groupby('MDDate')[
            'TotalVolumeTrade'].transform('sum')
        result = temp_df.groupby('MDDate')[['ClosePx', 'TotalVolumeTrade']].corr().unstack().iloc[:, 1]
        result = result.rolling(self.T, min_periods=1).mean()
        result = result.iloc[self.T - 1:]
        print(stock_name, self, 'finished!')
        return stock_name, result.index.values, result.values


class AvgOutFlowPropFactor(Factor):
    """
    平均单笔流出金额占比
    """

    def __init__(self, T: int):
        self.T = T
        self.factor_name = 'AvgOutFlowPropFactor'

    def __str__(self):
        return 'AvgOutFlowPropFactor'

    @func_time
    def cal(self, underlying: Underlying):
        stock_name = underlying.name
        temp_df = underlying.data[['MDDate', 'ClosePx', 'NumTrades', 'TotalValueTrade']].copy()
        back_sr = temp_df['ClosePx'].shift(1)
        temp_df['return'] = (temp_df['ClosePx'] - back_sr) / back_sr
        temp_df['return'] = vector_sign(temp_df['return'])
        temp_df.replace(np.nan, 0, inplace=True)
        temp_df.drop('ClosePx', axis=1, inplace=True)
        temp_df['i_NumTrades'] = temp_df['NumTrades'] * temp_df['return']
        temp_df['i_TotalValueTrade'] = temp_df['TotalValueTrade'] * temp_df['return']
        # print(temp_df)
        result = temp_df.groupby('MDDate')[['NumTrades', 'TotalValueTrade', 'i_NumTrades', 'i_TotalValueTrade']].sum()
        result = (result['i_TotalValueTrade'] / result['i_NumTrades']) / (
                    result['TotalValueTrade'] / result['NumTrades'])
        result = result.rolling(self.T, min_periods=1).mean()
        result = result.iloc[self.T - 1:]
        print(stock_name, self, 'finished!')
        return stock_name, result.index.values, result.values


def sign(x):
    if x >= 0:
        result = 0
    else:
        result = 1
    return result


def tow_column_corr(df: pd.DataFrame):
    corr = np.corrcoef(df.iloc[:, 0], df.iloc[:, 1])
    # print(corr, df.index.values)
    return pd.DataFrame({'daily_corr': pd.Series(corr, index=[df.index.values])})


def read_daily_factor(path: str) -> dict:
    result = {}
    df = pd.read_csv(path)
    df = df.groupby('stock')
    for stock_name, stock_frame in df:
        # if stock_name == '000629.SZ':
        #     print(stock_name)
        result[stock_name] = stock_frame.drop('stock', axis=1)
    return result

vector_sign = np.vectorize(sign)

# %%
if __name__ == '__main__':
    root = 'intraday'
    used_col_list = ['HTSCSecurityID', 'MDDate', 'MDTime', 'OpenPx',
                     'ClosePx', 'HighPx', 'LowPx', 'NumTrades',
                     'TotalVolumeTrade', 'TotalValueTrade']
    data_loader = DataLoader(root, '20170101 092500000', '20211231 150000000', used_col_list)
    # print(data_loader.path_list)
    test_underlying = data_loader.filter('intraday/kline2/000629.SZ.pkl')
# %%
    # close_vol_prop = CloseVolPropFactor('143000000', '150000000', 10)
    # skewness_kurtosis = SkewnessKurtosisFactor(10)
    # downward_vola_prop = DownwardVolaPropFactor(10)
    # vol_price_corr = VolPriceCorrFactor(10)
    avg_out_flow_prop = AvgOutFlowPropFactor(1)
    time_section_prop_open = TimeSectionPropFactor(feature_name='TotalVolumeTrade', section_start='093000000', section_end='100000000')
    time_section_prop_close = TimeSectionPropFactor(feature_name='TotalVolumeTrade', section_start='143000000', section_end='150000000')

    time_section_return_avg_daily = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='avg', section_start='093000000', section_end='150000000', T=1, N=1)
    time_section_return_avg_open = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='avg', section_start='093000000', section_end='100000000', T=1, N=1)
    time_section_return_avg_close = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='avg', section_start='143000000', section_end='150000000', T=1, N=1)
    time_section_return_std_daily = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='std', section_start='093000000', section_end='150000000', T=1, N=1)
    time_section_return_std_open = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='std', section_start='093000000', section_end='100000000', T=1, N=1)
    time_section_return_std_close = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='std', section_start='143000000', section_end='150000000', T=1, N=1)
    time_section_return_skew_daily = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='skew', section_start='093000000', section_end='150000000', T=1, N=1)
    time_section_return_skew_open = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='skew', section_start='093000000', section_end='100000000', T=1, N=1)
    time_section_return_skew_close = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='skew', section_start='143000000', section_end='150000000', T=1, N=1)
    time_section_return_kurt_daily = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='kurt', section_start='093000000', section_end='150000000', T=1, N=1)
    time_section_return_kurt_open = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='kurt', section_start='093000000', section_end='100000000', T=1, N=1)
    time_section_return_kurt_close = TimeSectionStatisticFactor(feature_name='ClosePx', stat_method='kurt', section_start='143000000', section_end='150000000', T=1, N=1)

    external_dict = read_daily_factor(os.path.join(root, 'tquant_factors' + os.sep + 'free_float_shares.csv'))
    time_section_turnover_std_close = TimeSectionStatisticFactor(feature_name='TotalVolumeTrade', stat_method='std',
                                                                 section_start='143000000', section_end='150000000',
                                                                 T=1, N=0, external=external_dict, external_name='turnover')
    time_section_turnover_std_open = TimeSectionStatisticFactor(feature_name='TotalVolumeTrade', stat_method='std',
                                                                section_start='093000000', section_end='100000000',
                                                                T=1, N=0, external=external_dict,
                                                                external_name='turnover')
    time_section_price_turnover_corr_daily = TimeSectionPriceCorrFactor('TotalVolumeTrade', section_start='093000000',
                                                                        section_end='150000000',
                                                                        T=1, external=external_dict, external_name='turnover_corr')
    price_range_turnover_std_high = PriceRangeStatisticFactor('TotalVolumeTrade', 'std', 0.5, high=True, T=1, N=0,
                                                              external=external_dict, external_name='turnover')
    price_range_turnover_mean_high = PriceRangeStatisticFactor('TotalVolumeTrade', 'avg', 0.5, high=True, T=1, N=0,
                                                              external=external_dict, external_name='turnover')
    # close_vol_prop_test = close_vol_prop.cal(test_underlying)
    # skewness_kurtosis_test = skewness_kurtosis.cal(test_underlying)
    # downward_vola_prop_test = downward_vola_prop.cal(test_underlying)
    # vol_price_corr_test = vol_price_corr.cal(test_underlying)
    avg_out_flow_prop_test = avg_out_flow_prop.cal(test_underlying)
    time_section_prop_open_test = time_section_prop_open.cal(test_underlying)
    time_section_prop_close_test = time_section_prop_close.cal(test_underlying)

    time_section_return_avg_daily_test = time_section_return_avg_daily.cal(test_underlying)
    time_section_return_avg_open_test = time_section_return_avg_open.cal(test_underlying)
    time_section_return_avg_close_test = time_section_return_avg_close.cal(test_underlying)
    time_section_return_std_daily_test = time_section_return_std_daily.cal(test_underlying)
    time_section_return_std_open_test = time_section_return_std_open.cal(test_underlying)
    time_section_return_std_close_test = time_section_return_std_close.cal(test_underlying)
    time_section_return_skew_daily_test = time_section_return_skew_daily.cal(test_underlying)
    time_section_return_skew_open_test = time_section_return_skew_open.cal(test_underlying)
    time_section_return_skew_close_test = time_section_return_skew_close.cal(test_underlying)
    time_section_return_kurt_daily_test = time_section_return_kurt_daily.cal(test_underlying)
    time_section_return_kurt_open_test = time_section_return_kurt_open.cal(test_underlying)
    time_section_return_kurt_close_test = time_section_return_kurt_close.cal(test_underlying)

    time_section_turnover_std_close_test = time_section_turnover_std_close.cal(test_underlying)
    time_section_turnover_std_open_test = time_section_turnover_std_open.cal(test_underlying)
    time_section_price_turnover_corr_daily_test = time_section_price_turnover_corr_daily.cal(test_underlying)
    price_range_turnover_std_high_test = price_range_turnover_std_high.cal(test_underlying)
    price_range_turnover_mean_high_test = price_range_turnover_mean_high.cal(test_underlying)

    # print(pd.isna(close_vol_prop_test[2]).sum())
    # print(skewness_kurtosis_test[2])
    # print(downward_vola_prop_test[2])
    # print(pd.isna(vol_price_corr_test[2]).sum())
    # print(vol_price_corr_test)
    print(avg_out_flow_prop, avg_out_flow_prop_test[0], avg_out_flow_prop_test[2])
    print(time_section_prop_open, time_section_prop_open_test[0], time_section_prop_open_test[2])
    print(time_section_prop_close, time_section_prop_close_test[0], time_section_prop_close_test[2])

    print(time_section_return_avg_daily, time_section_return_avg_daily_test[0], time_section_return_avg_daily_test[2])
    print(time_section_return_avg_open, time_section_return_avg_open_test[0], time_section_return_avg_open_test[2])
    print(time_section_return_avg_close, time_section_return_avg_close_test[0], time_section_return_avg_close_test[2])
    print(time_section_return_std_daily, time_section_return_std_daily_test[0], time_section_return_std_daily_test[2])
    print(time_section_return_std_open, time_section_return_std_open_test[0], time_section_return_std_open_test[2])
    print(time_section_return_std_close, time_section_return_std_close_test[0], time_section_return_std_close_test[2])
    print(time_section_return_skew_daily, time_section_return_skew_daily_test[0], time_section_return_skew_daily_test[2])
    print(time_section_return_skew_open, time_section_return_skew_open_test[0], time_section_return_skew_open_test[2])
    print(time_section_return_skew_close, time_section_return_skew_close_test[0], time_section_return_skew_close_test[2])
    print(time_section_return_kurt_daily, time_section_return_kurt_daily_test[0], time_section_return_kurt_daily_test[2])
    print(time_section_return_kurt_open, time_section_return_kurt_open_test[0], time_section_return_kurt_open_test[2])
    print(time_section_return_kurt_close, time_section_return_kurt_close_test[0], time_section_return_kurt_close_test[2])

    print(time_section_turnover_std_close_test, time_section_turnover_std_close_test[0],
          time_section_turnover_std_close_test[2])
    print(time_section_turnover_std_open_test, time_section_turnover_std_open_test[0], time_section_turnover_std_open_test[2])
    print(time_section_price_turnover_corr_daily_test, time_section_price_turnover_corr_daily_test[0], time_section_price_turnover_corr_daily_test[2])
    print(price_range_turnover_std_high_test, price_range_turnover_std_high_test[0], price_range_turnover_std_high_test[2])
    print(price_range_turnover_mean_high_test, price_range_turnover_mean_high_test[0], price_range_turnover_mean_high_test[2])

