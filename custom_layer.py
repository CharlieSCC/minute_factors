import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import itertools

def calc_std(tensor):
    x_std = torch.std(tensor,dim=1)
    #x_std = torch.switch(torch.is_nan(x_std), torch.mean(tensor, dim=1) - torch.mean(tensor, dim=1), x_std)
    if torch.sum(torch.isnan(x_std)):
        for i, bool in enumerate(torch.isnan(x_std)):
            if bool:
                new_std = torch.mean(tensor, dim=1) - torch.mean(tensor, dim=1)
                x_std[i] = new_std[i]
    return x_std



class ts_corr(nn.Module):
    def __init__(self, window=5, strides=1):
        super(ts_corr, self).__init__()
        self.strides = strides
        self.window = window

    def compute_corr(self, x, y):
        std_x = calc_std(x) + 0.00001
        std_y = calc_std(y) + 0.00001

        x_mul_y = x * y
        E_x_mul_y = torch.mean(x_mul_y, dim=1)
        mean_x = torch.mean(x, dim=1)
        mean_y = torch.mean(y, dim=1)
        cov = E_x_mul_y - mean_x * mean_y

        out = cov / (std_x * std_y)
        return out


    def forward(self, tensors):
        _, self.t_num, self.f_num, = tensors.shape
        self.s_num = 0
        self.c_num = int(self.f_num * (self.f_num - 1) / 2)
        xs = []

        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            for subset in itertools.combinations(list(range(self.f_num)), 2):
                tensor1 = tensors[:, i_stride:self.window + i_stride, subset[0]]
                tensor2 = tensors[:, i_stride:self.window + i_stride, subset[1]]
                x_corr = self.compute_corr(tensor1, tensor2)
                xs.append(x_corr)
            self.s_num += 1

        output = torch.stack(xs, dim=1)
        output = torch.reshape(output, (-1, self.s_num, self.c_num))
        return output.transpose(1,2)


class ts_cov(nn.Module):
    def __init__(self, window=5, strides=1):
        super(ts_cov, self).__init__()
        self.strides = strides
        self.window = window

    def compute_cov(self, x, y):
        x_mul_y = x * y
        E_x_mul_y = torch.mean(x_mul_y, dim=1)
        mean_x = torch.mean(x, dim=1)
        mean_y = torch.mean(y, dim=1)
        cov = E_x_mul_y - mean_x * mean_y
        return cov

    def forward(self, tensors):
        _, self.t_num, self.f_num, = tensors.shape
        self.s_num = 0
        self.c_num = int(self.f_num * (self.f_num - 1) / 2)
        xs = []

        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            for subset in itertools.combinations(list(range(self.f_num)), 2):
                tensor1 = tensors[:, i_stride:self.window + i_stride, subset[0]]
                tensor2 = tensors[:, i_stride:self.window + i_stride, subset[1]]
                x_corr = self.compute_cov(tensor1, tensor2)
                xs.append(x_corr)
            self.s_num += 1

        output = torch.stack(xs, dim=1)
        output = torch.reshape(output, (-1, self.s_num, self.c_num))
        return output.transpose(1,2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.s_num, self.c_num)


def calc_skew(tensor, eps):
    x_std = tensor - torch.mean(tensor, dim=1).unsqueeze(1)
    x_3 = torch.mean(x_std**3, dim=1)
    x_2 = torch.mean(x_std**2, dim=1) + eps

    x_skew = x_3/torch.pow(x_2, 1.5)
    #x_std = torch.switch(torch.is_nan(x_std), torch.mean(tensor, dim=1) - torch.mean(tensor, dim=1), x_std)
    if torch.sum(torch.isnan(x_skew)):
        for i, bool in enumerate(torch.isnan(x_skew)):
            if bool:
                new_skew = torch.mean(tensor, dim=1) - torch.mean(tensor, dim=1)
                x_skew[i] = new_skew[i]
    return x_skew

class ts_skew(nn.Module):
    def __init__(self, window=5, strides=1):
        super(ts_skew, self).__init__()
        self.strides = strides
        self.window = window
        self.eps = 0.000001

    def forward(self, tensors):
        _, self.t_num, self.f_num,  = tensors.shape
        self.s_num=0
        xs=[]
        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            for j in range(0,self.f_num):
                x_skew = calc_skew(tensors[:,i_stride:self.window+i_stride,j], self.eps)
                xs.append(x_skew)
            self.s_num += 1
        output = torch.stack(xs,dim=1)
        output = torch.reshape(output, (-1, self.s_num, self.f_num))
        return output.transpose(1,2)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.s_num , self.f_num)



def calc_kurt(tensor, eps):
    x_std = tensor - torch.mean(tensor, dim=1).unsqueeze(1)
    x_4 = torch.mean(x_std**4, dim=1)
    x_2 = torch.mean(x_std**2, dim=1) + eps

    x_kurt = x_4/torch.pow(x_2, 2)
    #x_std = torch.switch(torch.is_nan(x_std), torch.mean(tensor, dim=1) - torch.mean(tensor, dim=1), x_std)
    if torch.sum(torch.isnan(x_kurt)):
        for i, bool in enumerate(torch.isnan(x_kurt)):
            if bool:
                new_kurt = torch.mean(tensor, dim=1) - torch.mean(tensor, dim=1)
                x_kurt[i] = new_kurt[i]
    return x_kurt

class ts_kurt(nn.Module):
    def __init__(self, window=5, strides=1):
        super(ts_kurt, self).__init__()
        self.strides = strides
        self.window = window
        self.eps = 0.000001


    def forward(self, tensors):
        _, self.t_num, self.f_num,  = tensors.shape
        self.s_num=0
        xs=[]
        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            for j in range(0,self.f_num):
                x_kurt = calc_kurt(tensors[:,i_stride:self.window+i_stride,j], self.eps)
                xs.append(x_kurt)
            self.s_num += 1
        output = torch.stack(xs,dim=1)
        output = torch.reshape(output, (-1, self.s_num, self.f_num))
        return output.transpose(1,2)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.s_num , self.f_num)

class ts_std(nn.Module):
    def __init__(self, window=5,strides=1):
        super(ts_std, self).__init__()
        self.strides = strides
        self.window = window

    def forward(self, tensors):
        _, self.t_num, self.f_num,  = tensors.shape
        self.s_num=0
        xs=[]
        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            for j in range(0,self.f_num):
                x_std = calc_std(tensors[:,i_stride:self.window+i_stride,j])
                xs.append(x_std)
            self.s_num += 1
        output = torch.stack(xs,dim=1)
        output = torch.reshape(output, (-1, self.s_num, self.f_num))
        return output.transpose(1,2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.s_num , self.f_num)


class ts_zscore(nn.Module):
    def __init__(self, window=5, strides=1):
        super(ts_zscore, self).__init__()
        self.window = window
        self.strides = strides


    def forward(self, tensors):
        _, self.t_num, self.f_num,  = tensors.shape
        self.s = 0
        tmparray=[]
        def _df_ts_zscore(k):
            return ((tensors[:, self.window + k - 1,:]) - torch.mean(tensors[:, k:self.window + k,:], dim=1)) / (torch.std(tensors[:, k:self.window + k,:],dim=1)+1e-4)
        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            tmparray.append(_df_ts_zscore(i_stride))
        self.s=len(iter_list)
        output = torch.stack(tmparray,dim=1)
        output = torch.reshape(output, (-1, self.s, self.f_num))
        return output.transpose(1,2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.s, self.f_num)



class ts_decay_linear(nn.Module):
    def __init__(self, window=5,strides=1):
        super(ts_decay_linear, self).__init__()
        self.window = window
        self.strides = strides

    def forward(self, tensors):
        num = torch.reshape(torch.tensor(list(range(self.window))) + 1.0,(-1,1))
        coe = torch.tile(num, (1,tensors.shape[2])).to(tensors.device)
        self.s=0
        def _sub_decay_linear(k, coe):
            data = tensors[:, k:self.window + k, :]
            sum_days = torch.reshape(torch.sum(coe,dim = 0),(-1,tensors.shape[2]))
            sum_days = torch.tile(sum_days,(self.window,1)).to(tensors.device)
            coe = coe/sum_days
            decay = torch.sum(coe*data,dim = 1)
            return decay
        _, self.t_num, self.f_num,  = tensors.shape
        tmparray=[]
        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            tmparray.append(_sub_decay_linear(i_stride, coe))
        self.s=len(iter_list)
        output = torch.stack(tmparray,dim=1)
        output = torch.reshape(output, (-1, self.s, self.f_num))
        return output.transpose(1,2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.s, self.f_num)



class ts_return(nn.Module):
    def __init__(self, window=5,strides=1):
        super(ts_return, self).__init__()
        self.window = window
        self.strides = strides

    def forward(self, tensors):
        _, self.t_num, self.f_num,  = tensors.shape
        self.s = 0
        tmparray=[]
        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            tmparray.append((tensors[:, self.window + i_stride - 1] - tensors[:, i_stride]) / (tensors[:, i_stride]+1e-4).to(tensors.device))
        self.s = len(iter_list)
        output = torch.stack(tmparray,dim=1)
        output = torch.reshape(output, (-1, self.s, self.f_num))
        return output.transpose(1,2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.s, self.f_num)



class ts_mean(nn.Module):
    def __init__(self, window=5,strides=1):
        super(ts_mean, self).__init__()
        self.window = window
        self.strides = strides

    def forward(self, tensors):
        _, self.t_num, self.f_num,  = tensors.shape
        self.s=0
        tmparray=[]
        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            tmparray.append(torch.mean(tensors[:, i_stride:self.window + i_stride],dim=1))
        self.s=len(iter_list)
        output = torch.stack(tmparray, dim=1)
        output = torch.reshape(output, (-1, self.s, self.f_num))
        return output.transpose(1, 2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.s, self.f_num)




class ts_sum(nn.Module):
    def __init__(self, window=5, strides=1):
        super(ts_sum, self).__init__()
        self.window = window
        self.strides = strides

    def forward(self, tensors):
        _, self.t_num, self.f_num,  = tensors.shape
        self.s_num=0
        tmparray=[]
        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            tmparray.append(torch.sum(tensors[:, i_stride:self.window + i_stride - 1],dim=1))
        self.s_num = len(iter_list)
        output = torch.stack(tmparray,dim=1)
        output = torch.reshape(output, (-1, self.s_num, self.f_num))
        return output.transpose(1,2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.t_num - self.window+1 , self.f_num)



class ts_max(nn.Module):
    def __init__(self, window=5,strides=1):
        super(ts_max, self).__init__()
        self.window = window
        self.strides = strides

    def forward(self, tensors):
        _, self.t_num, self.f_num,  = tensors.shape
        self.s=0
        tmparray=[]
        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            tmparray.append(torch.max(tensors[:, i_stride:self.window + i_stride],dim=1)[0])
        self.s=len(iter_list)
        output = torch.stack(tmparray,dim=1)
        output = torch.reshape(output, (-1, self.s, self.f_num))
        return output.transpose(1,2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.s, self.f_num)



class ts_min(nn.Module):
    def __init__(self, window=5,strides=1):
        super(ts_min, self).__init__()
        self.window = window
        self.strides = strides

    def forward(self, tensors):
        _, self.t_num, self.f_num,  = tensors.shape
        self.s=0
        tmparray=[]
        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            tmparray.append(torch.min(tensors[:, i_stride:self.window + i_stride],dim=1)[0])
        self.s=len(iter_list)
        output = torch.stack(tmparray,dim=1)
        output = torch.reshape(output, (-1, self.s, self.f_num))
        return output.transpose(1,2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.s, self.f_num)


class ts_pct(nn.Module):
    def __init__(self, window=5, strides=1):
        super(ts_pct, self).__init__()
        self.strides = strides
        self.window = window

    def forward(self, inputs):
        tensors = inputs[:,:,:-2]
        indactor = inputs[:,:,-2]
        
        _, self.t_num, self.f_num  = tensors.shape
        self.s_num=0
        xs=[]
        iter_list = list(range(0, self.t_num - self.window + 1, self.strides))
        if self.t_num - self.window not in iter_list:
            iter_list.append(self.t_num - self.window)
        for i_stride in iter_list:
            for j in range(0,self.f_num):
                x_ = tensors[:,i_stride:self.window+i_stride,j] * indactor[:,i_stride:self.window+i_stride]
                x_ = torch.sum(x_, axis=1)/(torch.sum(torch.abs(tensors[:,i_stride:self.window+i_stride,j]), axis=1) + 0.00001)
                xs.append(x_)
            self.s_num += 1
        output = torch.stack(xs,dim=1)
        output = torch.reshape(output, (-1, self.s_num, self.f_num))
        return output.transpose(1,2)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.s_num , self.f_num)


class ts_pct30(ts_pct):
    def __init__(self, **kwargs):
        super(ts_pct30, self).__init__(5, 30)


class ts_corr30(ts_corr):
    def __init__(self, **kwargs):
        super(ts_corr30, self).__init__(5, 30)


class ts_cov30(ts_cov):
    def __init__(self, **kwargs):
        super(ts_cov30, self).__init__(5, 30)


class ts_std30(ts_std):
    def __init__(self, **kwargs):
        super(ts_std30, self).__init__(5, 30)


class ts_decay_linear30(ts_decay_linear):
    def __init__(self, **kwargs):
        super(ts_decay_linear30, self).__init__(5, 30)


class ts_zscore30(ts_zscore):
    def __init__(self, **kwargs):
        super(ts_zscore30, self).__init__(5, 30)


class ts_return30(ts_return):
    def __init__(self, **kwargs):
        super(ts_return30, self).__init__(5, 30)


class ts_mean30(ts_mean):
    def __init__(self, **kwargs):
        super(ts_mean30, self).__init__(5, 30)


class ts_sum30(ts_sum):
    def __init__(self, **kwargs):
        super(ts_sum30, self).__init__(5, 30)


class ts_max30(ts_max):
    def __init__(self, **kwargs):
        super(ts_max30, self).__init__(5, 30)


class ts_min30(ts_min):
    def __init__(self, **kwargs):
        super(ts_min30, self).__init__(5, 30)        

class ts_skew30(ts_skew):
    def __init__(self, **kwargs):
        super(ts_skew30, self).__init__(5, 30)   

class ts_kurt30(ts_kurt):
    def __init__(self, **kwargs):
        super(ts_kurt30, self).__init__(5, 30)   