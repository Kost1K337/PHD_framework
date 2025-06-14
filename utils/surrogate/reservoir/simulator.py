import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import re

from mipt_solver.ima.const import *


class Pconv(nn.Module):
    def __init__(self, *args, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(*args, kernel_size=kernel_size, padding=0, **kwargs)
        self.ks = kernel_size

    def forward(self, x):
        x = F.pad(x, [self.ks-1, 0])
        x = self.conv(x)
        return x

class Net(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.conv = nn.Sequential(
            Pconv(2, 4, kernel_size=20, stride=1),
            nn.ELU(),
            Pconv(4, 8, kernel_size=20, stride=1),
            nn.ELU(),
            Pconv(8, i, kernel_size=20, stride=1)
        ) 

    def forward(self, x):
        return self.conv(x)

class LinearRelu(nn.Module):
    def __init__(self, i):
        super().__init__()
        weights = torch.ones(i)
        #weights[:] = 1
        self._weights2 =  torch.ones(1,i)
        self._weights = nn.Parameter(weights)
        self._activation = nn.ReLU()
    def forward(self, x):
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        return (x @ self._weights).reshape(*x.shape[:-1], 1)

class RatesNet(nn.Module):
    def __init__(self, i, n ):
        super().__init__()
        self.layers = nn.Sequential(
          LinearRelu(i),
          nn.Linear(1, n),
          nn.ELU(),
          nn.Linear(n, 2)
        )


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class Simulator:
    well_list = ['601', '606', '701', '702', '704', '720']
    def __init__(self):
        self._pressure_models = { 
        }
        self._initial_pressures = {
        }
        self._rates_models = {
        }
        self._rates_models_parameters = {
        }
        self._rs_mean = {}
        for well in self.well_list:
            with open(f'weights_20_03/weights2/pressure_models/initial_pressures/{well}.pickle', 'rb') as f:
                self._initial_pressures[well] = pickle.load(f)
            # import pdb; pdb.set_trace()
            self._pressure_models[well] = Net(self._initial_pressures[well].size)
            self._pressure_models[well].load_state_dict(torch.load(f'weights_20_03/weights2/pressure_models/weights/{well}.pt'))
            with open(f'weights_20_03/weights2/rates_model/parameters/{well}.pickle', 'rb') as f:
                self._rates_models_parameters[well] = pickle.load(f)
            self._rates_models[well] = RatesNet(self._initial_pressures[well].size,
                                                self._rates_models_parameters[well]['n_neurons'])
            self._rates_models[well].load_state_dict(torch.load(f'weights_20_03/weights2/rates_model/weights/{well}.pt'))
            with open(f'weights_20_03/weights2/rates_model/rs_mean/{well}.pickle', 'rb') as f:
                self._rs_mean[well] = pickle.load(f)

    def __call__(self, bhps, history, injection_total):
        res={}
        well_dfs = dict([d for d in history.groupby(WELL_COL)])
        
        
        for well, bhp in bhps.items():
            if well_dfs.keys():
                
                inp = np.stack((np.hstack((well_dfs[well][WBHP_COL].values.reshape(-1), [bhp])),
                               np.hstack((well_dfs[well][INJT_COL].values.reshape(-1), [injection_total]))))
                inp[0, 0] = self._initial_pressures[well].mean()
                inp = torch.diff(torch.tensor(inp.astype(float)).float())

            else:
                inp = torch.diff(torch.Tensor([[0,],[0,]]))
            # import pdb; pdb.set_trace()
            pressure = self._pressure_models[well](inp)
            pressure = torch.cumsum(torch.cat(
                (torch.Tensor(self._initial_pressures[well]).reshape(-1,1), pressure), dim=-1), -1).detach().numpy()[:, -1:]
            # import pdb; pdb.set_trace()
            inp2 = pressure - bhp
            inp2 = (inp2 - self._rates_models_parameters[well]['features_mean'].reshape(-1,1)) / self._rates_models_parameters[well]['features_std'].reshape(-1,1)
            inp2 = torch.Tensor(inp2.T)
            # import pdb; pdb.set_trace()
            out = self._rates_models[well](inp2).detach().numpy()

            out = out * self._rates_models_parameters[well]['target_std'] + self._rates_models_parameters[well]['target_mean']
            
            
            res[well] = {
                QOIL_COL: out[0,0],
                QWAT_COL: out[0, 1],
                QGAS_COL: out[0, 0] * self._rs_mean[well],
                PRES_COL: pressure.mean().item()
            }
            
        return res
