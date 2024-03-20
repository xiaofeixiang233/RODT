# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : ${2023/10/27} ${12:28}
# @Author : Lizhan Hong, Helin Gong
# @Email : lzhong2048@sjtu.edu.cn
# @Software: ${RODT}
# @Lab: AISEA PLATFORM
# Input: Raw data set
# Output: Data set been preprocessed.

import os
import numpy as np
from sklearn.model_selection import train_test_split

# hyperparameter:
test_size = 0.1

# define the output names:
path_mu_train = 'DataPrepro/muTrain.txt'
path_mu_test = 'DataPrepro/muTest.txt'
path_alpha_train = 'DataPrepro/alphaTrain.txt'
path_alpha_test = 'DataPrepro/alphaTest.txt'

# load dataraw:
input_data_mu_nor = np.loadtxt('RawData/InpowerNor.txt')
input_data_alpha = np.loadtxt('RawData/Powercoef50.txt').T


def split_data_set():
    if os.path.exists('DataPrepro/muTrain.txt'):
        raise SystemError('You should generate the split once for all end2end methods')
    else:
        mu_train, mu_test, alpha_train, alpha_test = train_test_split(input_data_mu_nor, input_data_alpha,
                                                                      random_state=42, shuffle=True,
                                                                      test_size=test_size)

        np.savetxt(path_mu_train, mu_train)
        np.savetxt(path_mu_test, mu_test)
        np.savetxt(path_alpha_train, alpha_train)
        np.savetxt(path_alpha_test, alpha_test)
