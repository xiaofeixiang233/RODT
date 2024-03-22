import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error
from UI import TestdataIAEA
from typing import Union

def load_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"file {path} not found")
    return np.loadtxt(path)

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"model {path} not found")
    model = joblib.load(path)
    if model is None:
        raise ValueError(f"failed to load model from {path}") 
    print(f'load model from {path}')
    return model

def to_abs_path(path):
    if not os.path.isabs(path):
        return os.path.join(os.getcwd(), path)
    return path

class RODTModel:
    def __init__(self, name) -> None:
        self.name = name

        self.model_for = None
        self.model_inv = None

        self.model_dir = 'Models'

    def _get_model_for(self):
        raise NotImplementedError("get_model_for not implemented")

    def _get_model_inv(self):
        raise NotImplementedError("get_model_inv not implemented")    

    def load_model_for(self, model_path):
        self.model_for = load_model(to_abs_path(model_path))
    
    def load_model_inv(self, model_path):
        self.model_inv = load_model(to_abs_path(model_path))

    @staticmethod
    def _save_model(model, model_save_dir, model_name, name):
        model_save_dir = to_abs_path(model_save_dir)
        model_save_path = os.path.join(model_save_dir, name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        model_save_path = os.path.join(model_save_path, model_name)
        joblib.dump(model, model_save_path)
        return model_save_path

    def train_for(self, train_data_x, train_data_y, model_save_dir=None):
        if len(train_data_x.shape) != 2 or len(train_data_y.shape) != 2:
            raise ValueError("wrong data format : data_x and data_y should be 2D array")
        if model_save_dir is None:
            model_save_dir = self.model_dir

        self.model_for = self._get_model_for()
        self.model_for.fit(train_data_x, train_data_y)

        model_save_path = self._save_model(self.model_for, model_save_dir, f'model_for_{self.name}.joblib', self.name)
        print(f'train finished, save forward model to {model_save_path}')

    def train_inv(self, train_data_x, train_data_y, model_save_dir=None):
        if len(train_data_x.shape) != 2 or len(train_data_y.shape) != 2:
            raise ValueError("wrong data format : data_x and data_y should be 2D array")
        if model_save_dir is None:
            model_save_dir = self.model_dir

        self.model_inv = self._get_model_inv()
        self.model_inv.fit(train_data_x, train_data_y)

        model_save_path = self._save_model(self.model_inv, model_save_dir, f'model_inv_{self.name}.joblib', self.name)
        print(f'train finished, save inverse model to {model_save_path}')
    
    def _predict_for(self, input):
        return self.model_for.predict(input)
    
    def _predict_inv(self, input):
        return self.model_inv.predict(input)

    def eval_for(self, eval_data_x, eval_data_y):
        if self.model_for is None:
            raise ValueError("model_for not initialized")
        predict = self._predict_for(eval_data_x)
        print(f'MSE of forward model of {self.name}: {mean_squared_error(eval_data_y, predict)}')
    
    def eval_inv(self, eval_data_x, eval_data_y):
        if self.model_inv is None:
            raise ValueError("model_inv not initialized")
        predict = self._predict_inv(eval_data_x)
        print(f'MSE of inverse model of {self.name}: {mean_squared_error(eval_data_y, predict)}')

    def predict_for(self, data, test_index : int, basis : Union[str, np.ndarray], 
                    output_dir='Outputs', model_path=None, plot=True):
        if model_path is None:
            model_path = os.path.join(self.model_dir, self.name, f'model_for_{self.name}.joblib')
        if self.model_for is None:
            raise ValueError("model_for not load")
        if isinstance(basis, str):
            basis = load_file(os.path.join(os.getcwd(), basis))        
        
        output_path = to_abs_path(output_dir)
        output_path = os.path.join(output_path, self.name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, f'FieldPredicted_{self.name}.txt')
        
        alpha_predicted = self._predict_for(data[test_index, :].reshape(1, -1)).reshape(1, -1)
        field_predicted = alpha_predicted @ basis.T
        np.savetxt(output_path, field_predicted)
        print(f'save field prediction to {output_path}')

        if plot:
            self.plot3d(field_predicted, save_dir=output_dir)

    def predict_inv(self, data, test_index : int, scaling_norm : Union[str, np.ndarray], 
                    output_dir='Outputs', model_path=None, plot=True):
        if model_path is None:
            model_path = os.path.join(self.model_dir, self.name, f'model_inv_{self.name}.joblib')
        if self.model_inv is None:
            raise ValueError("model_inv not load")
        if isinstance(scaling_norm, str):
            scaling_norm = load_file(os.path.join(os.getcwd(), scaling_norm))    

        output_path = to_abs_path(output_dir)
        output_path = os.path.join(output_path, self.name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, f'MuPredicted_{self.name}.txt')
        
        mu_predicted = self._predict_inv(data[test_index, :].reshape(1, -1)).reshape(1, -1) @ np.linalg.pinv(scaling_norm)
        np.savetxt(output_path, mu_predicted)
        print(f'save mu prediction to {output_path}')

        if plot:
            raise NotImplementedError("plot for inverse prediction not implemented")
    
    def plot3d(self, data : Union[str, np.ndarray], path_control='RawData/index.xlsx', save_dir='Outputs'):
        path_control = to_abs_path(path_control)
        if not os.path.exists(path_control):
            raise FileNotFoundError(f"path control {path_control} not found")
        save_dir = to_abs_path(save_dir)
        save_dir = os.path.join(save_dir, self.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'Plot3d_{self.name}.png')
        TestdataIAEA(data_path=data, pathControl=path_control, pathSave=save_path)