# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:55:58 2020

@author: manai
"""

import numpy as np


# 0-1标准化
# 特征数据
def auto_norm(data_set):
    # 0代表求数据集一列中的最小值
    min_vals = data_set.min(0)
    # 0代表求数据集一列中的最大值
    max_vals = data_set.max(0)
    # shape返回行列数
    norm_data_set = np.zeros(data_set.shape)
    norm_data_set = (data_set - min_vals) / (max_vals - min_vals)
    return norm_data_set
