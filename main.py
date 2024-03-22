# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : ${2023/10/27} ${12:28}
# @Author : Lizhan Hong, Helin Gong
# @Email : lzhong2048@sjtu.edu.cn
# @Software: ${RODT}
# @Lab: AISEA PLATFORM

# from UI import *
# from KNN import *
# from SplitDataSet import *
# from NN import *
# from DT import *
# from SVM import *

from DT import DTModel
from KNN import KNNModel
from GBDT import GBDTModel
from SVM import SVMModel
import numpy as np

path_mu_train = 'DataPrepro/muTrain.txt'
path_mu_test = 'DataPrepro/muTest.txt'
path_alpha_train = 'DataPrepro/alphaTrain.txt'
path_alpha_test = 'DataPrepro/alphaTest.txt'
path_scaling_nor = 'RawData/scalingNor.txt'

basis = np.loadtxt('RawData/Powerbasis50.txt')
sensors = np.loadtxt('RawData/sensors.txt')

alpha_test = np.loadtxt(path_alpha_test)
alpha_train = np.loadtxt(path_alpha_train)
mu_test_nor = np.loadtxt(path_mu_test)
mu_train_nor = np.loadtxt(path_mu_train)
obs_train = alpha_train @ basis.T @ sensors.T
obs_test = alpha_test @ basis.T @ sensors.T

if __name__ == '__main__':
    # ==========================================================================
    # split the data set and don't run it again if you have splited one.
    # split_data_set()
    #
    # ==========================================================================
    # trainKNN()
    # predictKNN(num_test=10)

    # plot_result(choice='knn')
    # ==========================================================================
    #
    # ==========================================================================
    # nn = NN()
    # nn.train_for()
    # nn.train_inv()
    # nn.predict(num_test=10)

    # plot_result(choice='nn')
    # ==========================================================================
    #
    # ==========================================================================
    # trainDT()
    # predictDT(num_test=10)
    # plot_result(choice='dt')
    # ==========================================================================

    # ==========================================================================
    # trainSVM()
    # predictSVM(num_test=10)
    # plot_result(choice='svm')
    # ==========================================================================


    # ==========================================================================
    # dt_model = DTModel()
    # dt_model.train_for(mu_train_nor, alpha_train)
    # dt_model.eval_for(mu_test_nor, alpha_test)
    # dt_model.predict_for(mu_test_nor, 10, basis, plot=True)
    # ==========================================================================

    # ==========================================================================
    # knn_model = KNNModel()
    # knn_model.train_for(mu_train_nor, alpha_train)
    # knn_model.eval_for(mu_test_nor, alpha_test)
    # knn_model.predict_for(mu_test_nor, 10, basis, plot=True)
    # ==========================================================================


    # ==========================================================================
    # gbdt_model = GBDTModel()
    # gbdt_model.train_for(mu_train_nor, alpha_train)
    # gbdt_model.eval_for(mu_test_nor, alpha_test)
    # gbdt_model.predict_for(mu_test_nor, 10, basis, plot=True)
    # ==========================================================================

    
    # ==========================================================================
    svm_model = SVMModel(50, 4)
    svm_model.train_for(mu_train_nor, alpha_train)
    svm_model.load_model_for('Models/SVM', 'svm_mu_alpha')
    svm_model.eval_for(mu_test_nor, alpha_test)
    svm_model.predict_for(mu_test_nor, 10, basis, plot=True)
    
    # ==========================================================================