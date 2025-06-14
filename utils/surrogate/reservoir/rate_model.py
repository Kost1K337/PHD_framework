import numpy as np
import torch

import pickle
from .sat_simulator import Net, RatesNet
from .const import *
        
class Simulator:
    pass
"""     well_list = ['601', '606', '701', '702', '704', '720']
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
            with open(f'weights_01_02/pressure_models/initial_pressures/{well}.pickle', 'rb') as f:
                self._initial_pressures[well] = pickle.load(f)
            self._pressure_models[well] = Net(self._initial_pressures[well].size)
            self._pressure_models[well].load_state_dict(torch.load(f'weights_01_02/pressure_models/weights/{well}.pt'))
            with open(f'weights_01_02/rates_model/parameters/{well}.pickle', 'rb') as f:
                self._rates_models_parameters[well] = pickle.load(f)
            self._rates_models[well] = RatesNet(self._initial_pressures[well].size,
                                                self._rates_models_parameters[well]['n_neurons'])
            self._rates_models[well].load_state_dict(torch.load(f'weights_01_02/rates_model/weights/{well}.pt'))
            with open(f'weights_01_02/rates_model/rs_mean/{well}.pickle', 'rb') as f:
                self._rs_mean[well] = pickle.load(f)

    def __call__(self, bhps, history, injection_total):
        res={}
        well_dfs = dict([d for d in history.groupby(WELL_COL)])
        
        
        for well, bhp in bhps.items():
            if well_dfs.keys():
                
                inp = np.stack((np.hstack((well_dfs[well][WBHP_COL].values.reshape(-1), [bhp])),
                               np.hstack((well_dfs[well][INJT_COL].values.reshape(-1), [injection_total])))).astype(float)
                inp = torch.diff(torch.tensor(inp).float())
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
            out_raw = self._rates_models[well](inp2)
            out = out_raw.detach().numpy()

            out = out * self._rates_models_parameters[well]['target_std'] + self._rates_models_parameters[well]['target_mean']
            
            res[well] = {
                QOIL_COL: out[0,0],
                QWAT_COL: out[0, 1],
                QGAS_COL: out[0, 0] * self._rs_mean[well],
                PRES_COL: pressure.mean().item()
            }
            
        return res
"""