# -*- coding: utf-8 -*-
'''
@Project : min
@File : models.py
@Author : Shang Chencheng
@Date : 2022/8/22 13:12
'''
from custom_layer import *


class AlphaNet_extractor_30(nn.Module):
    def __init__(self,):
        super(AlphaNet_extractor_30, self).__init__()

        self.corr30 = ts_corr30()
        self.std30 = ts_std30()
        self.zscore30 = ts_zscore30()
        self.decay_linear30 = ts_decay_linear30()
        self.return30 = ts_return30()
        self.mean30 = ts_mean30()
        self.kurt30 = ts_kurt30()
        self.skew30 = ts_skew30()
        self.min30 = ts_min30()
        self.max30 = ts_max30()
        self.sum30 = ts_sum30()
        self.pct30_posi = ts_pct30()
        self.pct30_nega = ts_pct30()

    def forward(self, inputs):
        x = inputs[:, :, :-2]

        x_pct_posi30 = self.pct30_posi(inputs)
        x_pct_nega30 = self.pct30_nega(inputs)
        x_corr30 = self.corr30(x)
        # x_cov30 = self.cov30(x)
        x_std30 = self.std30(x)
        x_zscore30 = self.zscore30(x)
        x_decay30 = self.decay_linear30(x)
        # x_return30 = self.return30(x)
        x_kurt30 = self.kurt30(x)
        x_skew30 = self.skew30(x)
        x_min30 = self.min30(x)
        x_max30 = self.max30(x)
        x_sum30 = self.sum30(x)
        x_mean30 = self.mean30(x)
        x_add30 = torch.cat([x_pct_nega30,
                             x_pct_posi30,
                             x_corr30,
                             x_std30,
                             x_zscore30,
                             x_decay30,
                             x_min30,
                             x_max30,
                             x_sum30,
                             x_mean30,
                             x_skew30,
                             x_kurt30], dim=-2)
        return x_add30


class Transpose(nn.Module):
    def __init__(self,):
        super(Transpose, self).__init__()

    def forward(self, x):
        return x.transpose(1, 2)


class AlphaNet_min_intraday(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size):
        super(AlphaNet_min_intraday, self).__init__()
        self.extractor = AlphaNet_extractor_30()

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.feature_size_2_wise = int(
            self.feature_size * (self.feature_size - 1) / 2)

        self.GRU = nn.Sequential(
            nn.BatchNorm1d(self.feature_size_2_wise * 1 + self.feature_size * 11),
            Transpose(),
            nn.GRU(self.feature_size_2_wise * 1 + self.feature_size * 11, self.hidden_size, batch_first=True)
        )
        self.linear = nn.Linear(self.hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, load_flag=True):
        if load_flag:
            x_add30 = self.extractor(x.squeeze())
            output, _ = self.GRU(x_add30)
            x_out = self.linear(output[:, -1, :])

            return x_out.squeeze(), x_add30
        else:
            output, _ = self.GRU(x)
            x_out = self.linear(output[:, -1, :])
            return x_out.squeeze()


class AlphaNet_min_interday(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size):
        super(AlphaNet_min_interday, self).__init__()
        self.extractor = AlphaNet_extractor_30()

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.feature_size_2_wise = int(self.feature_size * (self.feature_size - 1) / 2)

        self.GRU = nn.Sequential(
            nn.BatchNorm1d(self.feature_size_2_wise * 1 + self.feature_size * 11),
            Transpose(),
            nn.GRU(self.feature_size_2_wise * 1 + self.feature_size * 11, self.hidden_size, batch_first=True)
        )
        self.linear = nn.Linear(hidden_size * 5, output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, load_flag=True):
        if load_flag:
            x_add30_0 = self.extractor(x[:, 0])
            x_add30_1 = self.extractor(x[:, 1])
            x_add30_2 = self.extractor(x[:, 2])
            x_add30_3 = self.extractor(x[:, 3])
            x_add30_4 = self.extractor(x[:, 4])

            output0, _ = self.GRU(x_add30_0)
            output1, _ = self.GRU(x_add30_1)
            output2, _ = self.GRU(x_add30_2)
            output3, _ = self.GRU(x_add30_3)
            output4, _ = self.GRU(x_add30_4)
            x_hidden = torch.cat([output0[:, -1, :], output1[:, -1, :], output2[:, -1, :], output3[:, -1, :], output4[:, -1, :]], dim=-1)
            x_out = self.linear(x_hidden)
            return x_out.squeeze(), torch.concat([x_add30_0.unsqueeze(-1), x_add30_1.unsqueeze(-1), x_add30_2.unsqueeze(-1), x_add30_3.unsqueeze(-1), x_add30_4.unsqueeze(-1)], dim=-1)
        else:
            output0, _ = self.GRU(x[:, :, :, 0])
            output1, _ = self.GRU(x[:, :, :, 0])
            output2, _ = self.GRU(x[:, :, :, 0])
            output3, _ = self.GRU(x[:, :, :, 0])
            output4, _ = self.GRU(x[:, :, :, 0])
            x_hidden = torch.cat([output0[:, -1, :], output1[:, -1, :], output2[:, -1, :], output3[:, -1, :], output4[:, -1, :]], dim=-1)
            x_out = self.linear(x_hidden)
            return x_out.squeeze()


