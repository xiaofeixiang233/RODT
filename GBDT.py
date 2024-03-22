'''
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
# hyperparameter:

# input path control
path_mu_train = 'DataPrepro/muTrain.txt'
path_mu_test = 'DataPrepro/muTest.txt'
path_alpha_train = 'DataPrepro/alphaTrain.txt'
path_alpha_test = 'DataPrepro/alphaTest.txt'

path_scaling_nor = 'RawData/scalingNor.txt'

# define the output names:
path_field_predicted = 'Outputs/FieldPredictedGBDT'
path_mu_predicted = 'Outputs/MuPredictedGBDT'
path_GBDT_model_for = 'Models/GBDTAlpha.pkl'
path_GBDT_model_inv = 'Models/GBDTY2Mu.pkl'

# load the data:
basis = np.loadtxt('RawData/Powerbasis50.txt')
sensors = np.loadtxt('RawData/sensors.txt')

alpha_test = np.loadtxt(path_alpha_test)
alpha_train = np.loadtxt(path_alpha_train)
mu_test_nor = np.loadtxt(path_mu_test)
mu_train_nor = np.loadtxt(path_mu_train)
obs_train = alpha_train @ basis.T @ sensors.T
obs_test = alpha_test @ basis.T @ sensors.T


def trainGBDT():
    # train the GBDT model
    # ===========================================================================================
    params_for={'n_estimators': 108, 'subsample': 0.35, 'max_depth': 48, 'learning_rate': 0.064, 'min_samples_leaf': 1, 'min_samples_split': 2, 'random_state': 42}
    model=GradientBoostingRegressor(**params_for)
    GBDT_for_model=MultiOutputRegressor(model)
    GBDT_for_model.fit(mu_train_nor, alpha_train)
    model=GradientBoostingRegressor(n_estimators=108,subsample=0.35,max_depth=49,learning_rate=0.060000000000000005,min_samples_leaf=1,min_samples_split=2,random_state=42)
    GBDT_inv_model=MultiOutputRegressor(model)
    GBDT_inv_model.fit(obs_train, mu_train_nor)
    

    # save the GBDT model:
    joblib.dump(GBDT_for_model, path_GBDT_model_for)
    joblib.dump(GBDT_inv_model, path_GBDT_model_inv)
    # ==========================================================================================


def predictGBDT(num_test=10):
    scalingNor = np.loadtxt(path_scaling_nor)

    # load models:
    model_for = joblib.load(path_GBDT_model_for)
    model_inv = joblib.load(path_GBDT_model_inv)

    # test on the test set
    alpha_predicted = model_for.predict(mu_test_nor[num_test, :].reshape(1, -1))
    field_predicted = alpha_predicted @ basis.T
    np.savetxt(path_field_predicted, field_predicted)

    Mu_predicted = model_inv.predict(obs_test[num_test, :].reshape(1, -1)) @ np.pinv(scalingNor)
    np.savetxt(path_mu_predicted, Mu_predicted)
'''

from basemodel import RODTModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

class GBDTModel(RODTModel):
    def __init__(self) -> None:
        super().__init__("GBDT")
        self.params_for={'n_estimators': 108, 'subsample': 0.35, 'max_depth': 48, 'learning_rate': 0.064, 
                         'min_samples_leaf': 1, 'min_samples_split': 2, 'random_state': 42}
        self.params_inv={'n_estimators': 108, 'subsample': 0.35, 'max_depth': 49, 'learning_rate': 0.060000000000000005, 
                         'min_samples_leaf': 1, 'min_samples_split': 2, 'random_state': 42}

    def _get_model_for(self):
        model = GradientBoostingRegressor(**self.params_for)
        return MultiOutputRegressor(model)

    def _get_model_inv(self):
        model = GradientBoostingRegressor(**self.params_inv)
        return MultiOutputRegressor(model)