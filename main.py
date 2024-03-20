# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : ${2023/10/27} ${12:28}
# @Author : Lizhan Hong, Helin Gong
# @Email : lzhong2048@sjtu.edu.cn
# @Software: ${RODT}
# @Lab: AISEA PLATFORM

from UI import *
from KNN import *
from SplitDataSet import *
from NN import *
from DT import *

if __name__ == '__main__':
    # ==========================================================================
    # split the data set and don't run it again if you have splited one.
    # split_data_set()
    #
    # ==========================================================================
    trainKNN()
    predictKNN(num_test=10)

    plot_result(choice='knn')
    # ==========================================================================
    #
    # ==========================================================================
    nn = NN()
    nn.train_for()
    nn.train_inv()
    nn.predict(num_test=10)

    plot_result(choice='nn')
    # ==========================================================================
    #
    # ==========================================================================
    trainDT()
    predictDT(num_test=10)
    plot_result(choice='dt')
    # ==========================================================================

