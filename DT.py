# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : ${2023/10/27} ${12:28}
# @Author : Lizhan Hong, Helin Gong
# @Email : lzhong2048@sjtu.edu.cn
# @Software: ${RODT}
# @Lab: AISEA PLATFORM
# Input: Parameter/Observation
# Output: State/BestParameter

'''
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# input path control
path_mu_train = 'DataPrepro/muTrain.txt'
path_mu_test = 'DataPrepro/muTest.txt'
path_alpha_train = 'DataPrepro/alphaTrain.txt'
path_alpha_test = 'DataPrepro/alphaTest.txt'

path_scaling_nor = 'RawData/scalingNor.txt'

# load the data:
basis = np.loadtxt('RawData/Powerbasis50.txt')
sensors = np.loadtxt('RawData/sensors.txt')

# define the output names:
path_field_predicted = 'Outputs/FieldPredictedDT'
path_mu_predicted = 'Outputs/MuPredictedDT'
path_dt_model_for = 'Models/mu_alpha.joblib'
path_dt_model_inv = 'Models/dt_y_mu.joblib'

alpha_test = np.loadtxt(path_alpha_test)
alpha_train = np.loadtxt(path_alpha_train)
mu_test_nor = np.loadtxt(path_mu_test)
mu_train_nor = np.loadtxt(path_mu_train)
obs_train = alpha_train @ basis.T @ sensors.T
obs_test = alpha_test @ basis.T @ sensors.T


def trainDT():
    model = DecisionTreeRegressor(max_depth=30, min_samples_leaf=1, min_samples_split=2)
    model.fit(mu_train_nor, alpha_train)

    predict = model.predict(mu_test_nor)
    print(mean_squared_error(alpha_test, predict))

    joblib.dump(model, path_dt_model_for)

    model = DecisionTreeRegressor(max_depth=30, min_samples_leaf=1, min_samples_split=2)
    model.fit(obs_train, mu_train_nor)
    joblib.dump(model, path_dt_model_inv)


def predictDT(num_test):
    scalingNor = np.loadtxt(path_scaling_nor)

    # load models:
    dt_model_for = joblib.load(path_dt_model_for)
    dt_model_inv = joblib.load(path_dt_model_inv)

    # test on the test set
    alpha_predicted = dt_model_for.predict(mu_test_nor[num_test, :].reshape(1, -1))
    field_predicted = alpha_predicted @ basis.T
    np.savetxt(path_field_predicted, field_predicted)

    Mu_predicted = dt_model_inv.predict(obs_test[num_test, :].reshape(1, -1)) @ np.linalg.pinv(scalingNor)
    np.savetxt(path_mu_predicted, Mu_predicted)
'''

from basemodel import RODTModel
from sklearn.tree import DecisionTreeRegressor

class DTModel(RODTModel):
    def __init__(self) -> None:
        super().__init__("DT")

    def _get_model_for(self):
        return DecisionTreeRegressor(max_depth=30, min_samples_leaf=1, min_samples_split=2)

    def _get_model_inv(self):
        return DecisionTreeRegressor(max_depth=30, min_samples_leaf=1, min_samples_split=2)