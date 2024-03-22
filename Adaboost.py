'''
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor

# hyperparameter:

# input path control
path_mu_train = 'DataPrepro/muTrain.txt'
path_mu_test = 'DataPrepro/muTest.txt'
path_alpha_train = 'DataPrepro/alphaTrain.txt'
path_alpha_test = 'DataPrepro/alphaTest.txt'

path_scaling_nor = 'RawData/scalingNor.txt'

# define the output names:
path_field_predicted = 'Outputs/FieldPredictedadaboost'
path_mu_predicted = 'Outputs/MuPredictedAdaboost'
path_adaboost_model_for = 'Models/adaboostAlpha.pkl'
path_adaboost_model_inv = 'Models/adaboostY2Mu.pkl'

# load the data:
basis = np.loadtxt('RawData/Powerbasis50.txt')
sensors = np.loadtxt('RawData/sensors.txt')

alpha_test = np.loadtxt(path_alpha_test)
alpha_train = np.loadtxt(path_alpha_train)
mu_test_nor = np.loadtxt(path_mu_test)
mu_train_nor = np.loadtxt(path_mu_train)
obs_train = alpha_train @ basis.T @ sensors.T
obs_test = alpha_test @ basis.T @ sensors.T


def trainAdaboost():
    # train the KNN model
    # ===========================================================================================
    model=DecisionTreeRegressor(
        max_depth=28,
        min_samples_leaf=1,
        min_samples_split=4,
        random_state=968)
    model = AdaBoostRegressor(estimator=model,
                              n_estimators=150,
                              learning_rate=0.385,
                              loss='square')
    adaboost_for_model=MultiOutputRegressor(model,n_jobs=5)
    adaboost_for_model.fit(mu_train_nor, alpha_train)
    model=DecisionTreeRegressor(
        max_depth=99,
        min_samples_leaf=1,
        min_samples_split=3,
        random_state=42)
    model = AdaBoostRegressor(estimator=model,
                              n_estimators=150,
                              learning_rate=0.385,
                              loss='square')
    adaboost_inv_model=MultiOutputRegressor(model,n_jobs=5)
    adaboost_inv_model.fit(obs_train, mu_train_nor)

    # save the KNN model:
    joblib.dump(adaboost_for_model, path_adaboost_model_for)
    joblib.dump(adaboost_inv_model, path_adaboost_model_inv)
    # ==========================================================================================


def predictAdaboost(num_test=10):
    scalingNor = np.loadtxt(path_scaling_nor)

    # load models:
    model_for = joblib.load(path_adaboost_model_for)
    model_inv = joblib.load(path_adaboost_model_inv)

    # test on the test set
    alpha_predicted = model_for.predict(mu_test_nor[num_test, :].reshape(1, -1))
    field_predicted = alpha_predicted @ basis.T
    np.savetxt(path_field_predicted, field_predicted)

    Mu_predicted = model_inv.predict(obs_test[num_test, :].reshape(1, -1)) @ np.pinv(scalingNor)
    np.savetxt(path_mu_predicted, Mu_predicted)
'''

from basemodel import RODTModel
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor

class AdaboostModel(RODTModel):
    def __init__(self):
        super().__init__("Adaboost")
    
    def _get_model_for(self):
        model=DecisionTreeRegressor(
            max_depth=28,
            min_samples_leaf=1,
            min_samples_split=4,
            random_state=968)
        model = AdaBoostRegressor(estimator=model,
                                n_estimators=150,
                                learning_rate=0.385,
                                loss='square')
        return MultiOutputRegressor(model,n_jobs=5)

    def _get_model_inv(self):
        model=DecisionTreeRegressor(
            max_depth=99,
            min_samples_leaf=1,
            min_samples_split=3,
            random_state=42)
        model = AdaBoostRegressor(estimator=model,
                                n_estimators=150,
                                learning_rate=0.385,
                                loss='square')
        return MultiOutputRegressor(model,n_jobs=5)
    