# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : ${2023/10/27} ${12:28}
# @Author : Lizhan Hong, Helin Gong
# @Email : lzhong2048@sjtu.edu.cn
# @Software: ${RODT}
# @Lab: AISEA PLATFORM
# Input: Parameter/Observation
# Output: State/BestParameter

import time
import os
import numpy as np
import warnings
import torch
import joblib
from torch import nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  ##如果没有这段代码，将会出现内核断掉的情况
warnings.filterwarnings("ignore")

# hyperparameters:
hidden_layers_for = [6, 6]
epochAdam_for = 2000
epochLBFGS_for = 2000
hidden_layers_inv = [6, 6]
epochAdam_inv = 2000
epochLBFGS_inv = 2000

# define the train/test data path
path_mu_train = 'DataPrepro/muTrain.txt'
path_mu_test = 'DataPrepro/muTest.txt'
path_alpha_train = 'DataPrepro/alphaTrain.txt'
path_alpha_test = 'DataPrepro/alphaTest.txt'

path_scaling_nor = 'RawData/scalingNor.txt'

# define the output names:
path_field_predicted = 'Outputs/FieldPredictedNN'
path_mu_predicted = 'Outputs/MuPredictedNN'
path_Stadard_Scalar_for = 'Models/NNSatadardScalarFor.pkl'
path_Stadard_Scalar_inv = 'Models/NNSatadardScalarInv.pkl'

path_nn_model_for = 'Models/adamLBFGSmodelFor'
path_nn_model_inv = 'Models/adamLBFGSmodelInv'

# load the data:
basis = np.loadtxt('RawData/Powerbasis50.txt')
sensors = np.loadtxt('RawData/sensors.txt')

alpha_test = np.loadtxt(path_alpha_test)
alpha_train = np.loadtxt(path_alpha_train)
mu_test_nor = np.loadtxt(path_mu_test)
mu_train_nor = np.loadtxt(path_mu_train)
obs_train = alpha_train @ basis.T @ sensors.T
obs_test = alpha_test @ basis.T @ sensors.T


class mNNn(nn.Module):
    def __init__(self, layers, data_path, device, xtrain, ytrain):
        super(mNNn, self).__init__()

        self.layers = layers
        self.data_path = data_path
        self.device = device
        self.xtrain = xtrain
        self.ytrain = ytrain

        self.weights, self.biases = self.initialize_NN(self.layers)

        self.loss_iter = []

        self.loss = None
        # self.loss_func = torch.nn.MSELoss(reduction='mean')
        self.loss_func = torch.nn.MSELoss()
        self.nIter = 0
        self.optimizer = None
        self.optimizer_name = None
        self.scheduler = None

        self.start_time = None

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)

        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = Variable(torch.zeros([1, layers[l + 1]],
                                     dtype=torch.float32)).to(self.device)
            b.requires_grad_()
            weights.append(W)
            biases.append(b)
        return weights, biases

    def detach(self, data):
        return data.detach().cpu().numpy()

    def xavier_init(self, size):
        W = Variable(nn.init.xavier_normal_(torch.empty(size[0], size[1]))).to(
            self.device)
        W.requires_grad_()
        return W

    def data_loader(self, x, requires_grad=True):
        x_tensor = torch.tensor(x,
                                requires_grad=requires_grad,
                                dtype=torch.float32)
        return x_tensor.to(self.device)

    # def coor_shift(self, X):
    #    X_shift = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
    #    return X_shift

    def neural_net(self, x, weights, biases):
        num_layers = len(weights) + 1
        # X = self.coor_shift(x)
        X = x
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            X = torch.relu(torch.add(torch.matmul(X, W), b))
            # relu,tanh
        W = weights[-1]
        b = biases[-1]
        Y = torch.add(torch.matmul(X, W), b)
        return Y

    def net_u(self, x):
        u = self.neural_net(x, self.weights, self.biases)
        return u

    def forward(self, x):
        u = self.net_u(x)
        return u.detach().cpu().numpy().squeeze()

    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()

        # Loss function initialization
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        self.loss.requires_grad_()

        prediction = self.net_u(self.xtrain)
        self.loss = self.loss_func(prediction, self.ytrain)  # must be (1. nn output, 2. target)

        self.loss.backward()
        self.nIter = self.nIter + 1

        loss_remainder = 100
        if np.remainder(self.nIter, loss_remainder) == 0:
            loss = self.detach(self.loss)
            log_str = str(self.optimizer_name) + ' Iter ' + str(self.nIter) + ' Loss : %.10f' % (loss)
            print(log_str)
            elapsed = time.time() - self.start_time
            print('Iter:', loss_remainder, 'Time: %.4f' % (elapsed))
            self.start_time = time.time()
        return self.loss

    # Some optimization algorithms such as Conjugate Gradient and LBFGS need to
    # reevaluate the function multiple times, so you have to pass in a closure
    # that allows them to recompute your model. The closure should clear the gradients,
    # compute the loss, and return it.
    # https://pytorch.org/docs/stable/optim.html
    def train_LBFGS(self, optimizer, LBFGS_scheduler):
        self.optimizer = optimizer
        self.optimizer_name = 'LBFGS'
        self.scheduler = LBFGS_scheduler

        def closure():
            loss = self.optimize_one_epoch()
            bloss = loss.detach().numpy()
            self.loss_iter.append(bloss)  # record the loss for each iteration
            if self.scheduler is not None:
                self.scheduler.step()
            return loss

        self.optimizer.step(closure)
        return self.loss_iter

    def train_Adam(self, optimizer, nIter, Adam_scheduler):
        self.optimizer = optimizer
        self.optimizer_name = 'Adam'
        self.scheduler = Adam_scheduler
        for it in range(nIter):
            loss = self.optimize_one_epoch()
            bloss = loss.detach().numpy()
            self.loss_iter.append(bloss)  # record the loss for each iteration
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(self.loss)
        return self.loss_iter

    def predict(self, X_input):
        x = self.data_loader(X_input)
        with torch.no_grad():
            u = self.forward(x)
            return u


class NN:
    def __init__(self):
        '''

        :param p: The distance constant of 'Minkovski Distance', 'p=1' for 'Manhattan Distance', 'p=2' for 'Euclidean Distance'
        :param k1:
        :param k2:
        :param r:
        '''
        # Data preprocessing
        self.ss = None
        self.sensors = sensors
        self.basis = basis

        self.scaling_index = np.loadtxt(path_scaling_nor)
        self.data_path = 'Models'

        self.mu_train = np.loadtxt(path_mu_train)

        self.alpha_train = np.loadtxt(path_alpha_train)

        self.mu_test = np.loadtxt(path_mu_test)

        self.alpha_test = np.loadtxt(path_alpha_test)

    def train_for(self):
        self.train(X_train=self.mu_train, X_test=self.mu_test, y_train=self.alpha_train, y_test=self.alpha_test,
                   outputmodel=path_nn_model_for,
                   outputloss=self.data_path + '/lossFor.txt',
                   outputsample=self.data_path + '/sampleFor.txt',
                   path_SS=path_Stadard_Scalar_for,
                   hidden_layers=hidden_layers_for,
                   epochAdam=epochAdam_for,
                   epochLBFGS=epochLBFGS_for)

    def train_inv(self):
        Y_train, Y_test = self.alpha_train @ self.basis.T @ self.sensors.T, self.alpha_test @ self.basis.T @ self.sensors.T
        self.train(X_train=Y_train, X_test=Y_test, y_train=self.mu_train, y_test=self.mu_test,
                   outputmodel=path_nn_model_inv,
                   outputloss=self.data_path + '/lossInv.txt',
                   outputsample=self.data_path + '/sampleInv.txt',
                   path_SS=path_Stadard_Scalar_inv,
                   hidden_layers=hidden_layers_inv,
                   epochAdam=epochAdam_inv,
                   epochLBFGS=epochLBFGS_inv)

    def train(self, X_train, X_test, y_train, y_test,
              path_SS,
              outputmodel, outputloss, outputsample,
              hidden_layers, epochAdam, epochLBFGS):
        # Use cuda
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        print('using device ' + device)
        device = torch.device(device)

        # -------------------------
        # X_train = self.X_train
        # y_train = self.y_train
        # X_test = self.X_test
        # y_test = self.y_test
        # -----------------------

        n_layer_0, n_layer_minus1 = X_train.shape[1], y_train.shape[1]

        # to StandardScaler data, i.e., average = 0; standard deviation = 1
        self.ss = StandardScaler()
        X_train = self.ss.fit_transform(X_train)
        X_test = self.ss.transform(X_test)
        joblib.dump(self.ss, path_SS)

        # to tensor form
        x = torch.tensor(X_train)
        y = torch.tensor(y_train)
        print(x.shape, y.shape)
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        layers = [n_layer_0] + hidden_layers +  [n_layer_minus1]
        # Set training Epoches for Adam
        N_Adam = epochAdam
        # Set training Epoches for LBFGS
        N_LBFGS = epochLBFGS
        # Load PINN

        model = mNNn(layers, self.data_path, device, x, y)
        model.to(device)

        # outputmodel = self.data_path + '/adamLBFGSmodelFor'
        solvemodel = os.path.exists(outputmodel)
        # solvemodel = 0 # set solvemodel to 0 to train the model, otherwise load the model

        if solvemodel:
            model = torch.load(outputmodel)  # load model
        else:
            Adam_optimizer = torch.optim.Adam(params=model.weights + model.biases,
                                              lr=2e-3,
                                              betas=(0.9, 0.999),
                                              eps=1e-8,
                                              weight_decay=0,
                                              amsgrad=False)
            model.train_Adam(Adam_optimizer, N_Adam, None)

            LBFGS_optimizer = torch.optim.LBFGS(params=model.weights + model.biases,
                                                lr=1,
                                                max_iter=N_LBFGS,
                                                tolerance_grad=-1,
                                                tolerance_change=-1,
                                                history_size=100,
                                                line_search_fn=None)
            loss = model.train_LBFGS(LBFGS_optimizer, None)

            # save the loss function of each iteration
            # outlossfile = self.data_path + '/lossFor.txt'
            myFile1 = open(outputloss, 'w+')
            a = np.array(loss)
            np.savetxt(myFile1, a)
            myFile1.close()

            # save model
            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
            torch.save(model, outputmodel)

        # # the model has been trained, use X_test for prediction
        # x = torch.tensor(X_test)
        # x = x.to(torch.float32)
        # prediction_test = model.predict(x)
        # if prediction_test.ndim == 1:
        #     prediction_test = prediction_test.reshape(-1, 1)  # N-vector to matrix N*1
        # z = prediction_test - y_test
        # result = np.hstack((prediction_test, y_test, z))
        #
        # # outputfile = data_path + 'output/sample.txt'
        # myFile = open(outputsample, 'w+')
        # print("The array is:", result)
        # np.savetxt(myFile, result)
        # myFile.close()

        # # save the latent-variables
        # save(data=model.latent_vars, path=str(self.data_name) + '/MLOutput/adamLBFGS/latentVars.pkl', choice='pickle')

    def predict(self, num_test):
        ss_for = joblib.load(path_Stadard_Scalar_for)
        ss_inv = joblib.load(path_Stadard_Scalar_inv)
        nn_model_for = torch.load(path_nn_model_for)
        nn_model_inv = torch.load(path_nn_model_inv)

        # For
        mu_test = mu_test_nor[num_test, :].reshape(1, -1)
        mu_test = ss_for.transform(mu_test)
        x_for = torch.tensor(mu_test)
        x_for = x_for.to(torch.float32)
        alpha_predicted = nn_model_for.predict(x_for)
        field_predicted = alpha_predicted @ basis.T
        np.savetxt(path_field_predicted, field_predicted)

        # Inv
        obs = obs_test[num_test,:].reshape(1, -1)
        obs = ss_inv.transform(obs)
        x_inv = torch.tensor(obs)
        x_inv = x_inv.to(torch.float32)
        Mu_predicted = nn_model_inv.predict(x_inv)
        Mu_predicted = Mu_predicted.reshape((1, -1))
        np.savetxt(path_mu_predicted, Mu_predicted)