# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : ${2023/10/27} ${12:28}
# @Author : Lizhan Hong, Helin Gong
# @Email : lzhong2048@sjtu.edu.cn
# @Software: ${RODT}
# @Lab: AISEA PLATFORM
# Input: Parameter/Observation
# Output: State/BestParameter


import numpy as np
import joblib
from sklearn.neighbors import KNeighborsRegressor

# hyperparameter:
p = 2
k_for = 5
k_inv = 1

# input path control
path_mu_train = 'DataPrepro/muTrain.txt'
path_mu_test = 'DataPrepro/muTest.txt'
path_alpha_train = 'DataPrepro/alphaTrain.txt'
path_alpha_test = 'DataPrepro/alphaTest.txt'

path_scaling_nor = 'RawData/scalingNor.txt'

# define the output names:
path_field_predicted = 'Outputs/FieldPredictedKNN'
path_mu_predicted = 'Outputs/MuPredictedKNN'
path_knn_model_for = 'Models/knnMu2Alpha.pkl'
path_knn_model_inv = 'Models/knnY2Mu.pkl'

# load the data:
basis = np.loadtxt('RawData/Powerbasis50.txt')
sensors = np.loadtxt('RawData/sensors.txt')

alpha_test = np.loadtxt(path_alpha_test)
alpha_train = np.loadtxt(path_alpha_train)
mu_test_nor = np.loadtxt(path_mu_test)
mu_train_nor = np.loadtxt(path_mu_train)
obs_train = alpha_train @ basis.T @ sensors.T
obs_test = alpha_test @ basis.T @ sensors.T


def trainKNN():
    # train the KNN model
    # ===========================================================================================
    knn_model_for = KNeighborsRegressor(n_neighbors=k_for, weights='distance', p=p, metric='minkowski')
    knn_model_for.fit(mu_train_nor, alpha_train)

    knn_model_inv = KNeighborsRegressor(n_neighbors=k_inv, weights='distance', p=p, metric='minkowski')
    knn_model_inv.fit(obs_train, mu_train_nor)

    # save the KNN model:
    joblib.dump(knn_model_for, path_knn_model_for)
    joblib.dump(knn_model_inv, path_knn_model_inv)
    # ==========================================================================================


def predictKNN(num_test=10):
    scalingNor = np.loadtxt(path_scaling_nor)

    # load models:
    knn_model_for = joblib.load(path_knn_model_for)
    knn_model_inv = joblib.load(path_knn_model_inv)

    # test on the test set
    alpha_predicted = knn_model_for.predict(mu_test_nor[num_test, :].reshape(1, -1))
    field_predicted = alpha_predicted @ basis.T
    np.savetxt(path_field_predicted, field_predicted)

    Mu_predicted = knn_model_inv.predict(obs_test[num_test, :].reshape(1, -1)) @ np.linalg.pinv(scalingNor)
    np.savetxt(path_mu_predicted, Mu_predicted)
