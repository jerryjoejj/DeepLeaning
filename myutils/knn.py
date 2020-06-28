# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:39:38 2020

@author: manai
"""

import numpy as np


def knn(in_x, data_set, labels, k):
    dist = (((data_set - in_x) ** 2).sum(1)) ** 0.5
    sorted_dist = dist.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    max_type = 0
    max_count = -1
    for key, value in class_count.items():
        if value > max_count:
            max_type = key
            max_count = value
    return max_type
