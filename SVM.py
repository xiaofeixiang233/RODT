'''
import numpy as np
import joblib
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

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
path_svm_field_predicted = 'Outputs/FieldPredictedSVM'
path_svm_mu_predicted = 'Outputs/MuPredictedSVM'
path_svm_model_for = 'Models/SVM/svm_mu_alpha'
path_svm_model_inv = 'Models/SVM/svm_y_mu'

alpha_test = np.loadtxt(path_alpha_test)
alpha_train = np.loadtxt(path_alpha_train)
mu_test_nor = np.loadtxt(path_mu_test)
mu_train_nor = np.loadtxt(path_mu_train)
obs_train = alpha_train @ basis.T @ sensors.T
obs_test = alpha_test @ basis.T @ sensors.T

print(alpha_train.shape, mu_train_nor.shape, obs_train.shape)

def trainSVM():
    # ———————————— forward model ————————————
    models = []
    # for i in range(alpha_train.shape[1]):
    pbar = tqdm(range(alpha_train.shape[1]))
    for i in pbar:
        pbar.set_description(f'Training model_for_output{i}')
        model = SVR(kernel="rbf", gamma=25, C=10.0, epsilon=0.02)  # gamma=25, C=10.0, epsilon=0.02
        model.fit(mu_train_nor, alpha_train[..., i])
        joblib.dump(model, path_svm_model_for + str(i) + '.joblib')
        models.append(model)

    mse = []
    for i, model in enumerate(models):
        mse_i = mean_squared_error(alpha_test[..., i], model.predict(mu_test_nor))
        mse.append(mse_i)

    print(f'alpha loss : {sum(mse) / len(mse)}')

    # ———————————— inv model ————————————
    models = []
    pbar = tqdm(range(mu_train_nor.shape[1]))
    for i in pbar:
        pbar.set_description(f'Training model_inv_output{i}')
        model = SVR(kernel="rbf", gamma=30, C=10.0, epsilon=0.02)  # gamma=25, C=10.0, epsilon=0.02
        model.fit(obs_train, mu_train_nor[..., i])
        joblib.dump(model, path_svm_model_inv + str(i) + '.joblib')
        models.append(model)

    mse = []
    for i, model in enumerate(models):
        mse_i = mean_squared_error(mu_test_nor[..., i], model.predict(obs_test))
        mse.append(mse_i)

    print(f'mu loss : {sum(mse) / len(mse)}')


def predictSVM(num_test):
    scalingNor = np.loadtxt(path_scaling_nor)

    # load models:
    svm_model_for = [joblib.load(path_svm_model_for + str(i) + ".joblib") for i in range(0, alpha_train.shape[1])]
    # svm_model_inv = joblib.load(path_svm_model_inv)

    # test on the test set
    alpha_predicted = [svm_model_for[i].predict(mu_test_nor[num_test, :].reshape(1, -1)) for i in range(0, alpha_train.shape[1])]
    alpha_predicted = np.concatenate(alpha_predicted, axis=0).reshape(1, -1)
    print(alpha_predicted.shape, basis.T.shape)
    field_predicted = alpha_predicted @ basis.T
    np.savetxt(path_svm_field_predicted, field_predicted)
        
    # alpha_predicted = svm_model_for.predict(mu_test_nor[num_test, :].reshape(1, -1))
    # field_predicted = alpha_predicted @ basis.T
    # np.savetxt(path_field_predicted, field_predicted)

    # Mu_predicted = dt_model_inv.predict(obs_test[num_test, :].reshape(1, -1)) @ np.linalg.pinv(scalingNor)
    # np.savetxt(path_mu_predicted, Mu_predicted)
'''

from basemodel import RODTModel, to_abs_path, load_model
from sklearn.svm import SVR
from tqdm import tqdm
import numpy as np
import os

class SVMModel(RODTModel):
    def __init__(self, no_for, no_inv) -> None:
        super().__init__("SVM")
        self.no_for = no_for  # number of ouputs for forward model
        self.no_inv = no_inv  # number of ouputs for inverse model

    def _get_model_for(self):
        return [SVR(kernel="rbf", gamma=25, C=10.0, epsilon=0.02) for i in range(self.no_for)]
    
    def _get_model_inv(self):
        return [SVR(kernel="rbf", gamma=30, C=10.0, epsilon=0.02) for i in range(self.no_inv)]

    def train_for(self, train_data_x, train_data_y, model_save_dir=None):
        if len(train_data_x.shape) != 2 or len(train_data_y.shape) != 2:
            raise ValueError("wrong data format : data_x and data_y should be 2D array")
        if train_data_y.shape[-1] != self.no_for:
            raise ValueError(f"wrong output number : output_num should be {train_data_y.shape[-1]}")
        if model_save_dir is None:
            model_save_dir = self.model_dir

        self.model_for = self._get_model_for()        
        pbar = tqdm(range(self.no_for))
        for i in pbar:
            pbar.set_description(f'Training model_for_{self.name}_{i}')
            self.model_for[i].fit(train_data_x, train_data_y[..., i])
            model_save_path = self._save_model(self.model_for[i], model_save_dir, f'model_for_{self.name}{i}.joblib', self.name)

        print('train finished, save forward model to', model_save_path.split('/')[:-1])

    def train_inv(self, train_data_x, train_data_y, model_save_dir=None):
        if len(train_data_x.shape) != 2 or len(train_data_y.shape) != 2:
            raise ValueError("wrong data format : data_x and data_y should be 2D array")
        if train_data_y.shape[-1] != self.no_inv:
            raise ValueError(f"wrong output number : output_num should be {train_data_y.shape[-1]}")
        if model_save_dir is None:
            model_save_dir = self.model_dir

        self.model_inv = self._get_model_inv()
        pbar = tqdm(range(self.no_inv))
        for i in pbar:
            pbar.set_description(f'Training model_inv_{self.name}_{i}')
            self.model_inv[i].fit(train_data_x, train_data_y[..., i])
            model_save_path = self._save_model(self.model_inv[i], model_save_dir, f'model_inv_{self.name}{i}.joblib', self.name)

        print('train finished, save inverse model to', model_save_path.split('/')[:-1])

    def _predict_for(self, input):
        results = [self.model_for[i].predict(input).reshape(-1, 1) for i in range(0, self.no_for)]
        return np.concatenate(results, axis=-1)

    def _predict_inv(self, input):
        results = [self.model_inv[i].predict(input).reshape(-1, 1) for i in range(0, self.no_inv)]
        return np.concatenate(results, axis=-1)
    
    def load_model_for(self, model_path, name='model_for_SVM'):
        self.model_for = [load_model(os.path.join(to_abs_path(model_path), f'{name}{i}.joblib')) for i in range(self.no_for)]

    def load_model_inv(self, model_path, name='model_inv_SVM'):
        self.model_inv = [load_model(os.path.join(to_abs_path(model_path), f'{name}{i}.joblib')) for i in range(self.no_inv)]
