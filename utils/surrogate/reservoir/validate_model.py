import torch as tc
import numpy as np
import os
import sys
import json
import model.multipolygon as mpg
from model.proxymodelmix import ProxyModelNetMix
from model.multimodelmix import MultiModelMix
import show

import matplotlib.pyplot as plt
import pandas as pd

class Predictor:
    def __init__(self, model, device=tc.device('cpu')):
        self.model = model
        self.device = device

    def _history_step(self, time_step: int, hidden_prev: tc.Tensor):
        if time_step > 0:
            hidden_cur = self.model.step(time_step - 1, hidden_prev)
        else:
            hidden_cur = self.model.start()

        self.model.save_hidden(time_step, hidden_cur)

        view = self.model.view(time_step, hidden_cur)
        self.model.save_view(time_step, view)

        return hidden_cur

    @tc.inference_mode()
    def make_history(self):
        time_steps = self.model.time_steps()

        hidden = None
        for time_step in range(time_steps):
            hidden = self._history_step(time_step, hidden)

def show_error(times, error_min, error_mean, error_max, name, path):
        fig, ax = plt.subplots(1, figsize=(19,10))

        ax.plot(times, error_min, label = 'error_min_' + name)
        ax.plot(times, error_mean, label = 'error_mean_' + name)
        ax.plot(times, error_max, label = 'error_max_' + name)
        ax.legend()

        plt.savefig(path)

def show_hist_error(error1, error2, error3, name1, name2, name3, n_bins, path):
        fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=True)

        axs[0].hist(error1.flatten(), bins=n_bins, density=True)#, label = 'error_' + name1)
        axs[1].hist(error2.flatten(), bins=n_bins, density=True)#, label = 'error_' + name2)
        axs[2].hist(error3.flatten(), bins=n_bins, density=True)#, label = 'error_' + name2)
        #axs.legend()

        plt.savefig(path)

def run(params_json_file):
    with open(params_json_file) as params_file:
        params = json.load(params_file)

    print(params)
    print('---------------------------------------------------')

    test_data_file = params['test_data_file']
    delimiter = params['delimiter']

    result_folder = params['result_folder']
    model_filename = params['model_filename']

    hidden_per_well = params['hidden_per_well']
    addition_cells_mult = params['addition_cells_per_well']

    device = tc.device(params['device'])
    tc.manual_seed(0)

    print('test data:', test_data_file)
    print('---------------------------------------------------')

    poly_test = mpg.MultPolygon()
    poly_test.read_history(test_data_file, delimiter=delimiter)

    well_count = poly_test.well_count
    addition_cells = int(np.round(well_count * addition_cells_mult))

    model = ProxyModelNetMix(well_count, 1.0, hidden_per_well, addition_cells)

    model.load_state_dict(tc.load(model_filename))
    model = model.to(device=device)

    offset_bhp, mult_bhp, offset_prod, mult_prod = model.get_scales()
    poly_test.scale(offset_bhp, mult_bhp, offset_prod, mult_prod)
    poly_predict = poly_test.copy()

    smodel_test = MultiModelMix(model, poly_predict, device=device)
    predictor = Predictor(smodel_test, device)

    predictor.make_history()

    poly_test.unscale(offset_bhp, mult_bhp, offset_prod, mult_prod)
    poly_predict.unscale(offset_bhp, mult_bhp, offset_prod, mult_prod)
    poly_predict.print_results(result_folder + "/result_valid.feth")

    df = pd.read_feather(result_folder + "/result_valid.feth")
    df.to_excel(result_folder + "/result_valid.xlsx")

    prod_pred = np.cumsum(poly_predict.prod_hist * poly_predict.wf_hist, axis=0)
    prod_test = np.cumsum(poly_test.prod_hist * poly_test.wf_hist, axis=0)

    bhp_pred = poly_predict.bhp_hist
    bhp_test = poly_test.bhp_hist

    oil_pred = np.cumsum(poly_predict.prod_hist * (1 - poly_predict.sat_hist) * poly_predict.wf_hist, axis=0).sum(axis=2)
    oil_test = np.cumsum(poly_test.prod_hist * (1 - poly_test.sat_hist) * poly_test.wf_hist, axis=0).sum(axis=2)

    print(oil_pred[-1,:])
    print(oil_test[-1,:])

    error_prod = np.abs(prod_pred - prod_test) / (np.abs(prod_test) + np.abs(prod_pred) + 1e-6) * 2.0
    error_bhp = np.abs(bhp_pred - bhp_test) / (np.abs(bhp_test) + np.abs(bhp_pred) + 1e-6) * 2.0
    error_oil =  np.abs(oil_pred - oil_test) / (np.abs(oil_test) + np.abs(oil_pred) + 1e-6) * 2.0

    error_prod = np.minimum(np.maximum(error_prod, 0), 1)
    error_bhp = np.minimum(np.maximum(error_bhp, 0), 1)
    error_oil = np.minimum(np.maximum(error_oil, 0), 1)

    error_prod_min = error_prod.min(axis=(1,2))
    error_prod_mean = error_prod.mean(axis=(1,2))
    error_prod_max = error_prod.max(axis=(1,2))

    error_bhp_min = error_bhp.min(axis=(1,2))
    error_bhp_mean = error_bhp.mean(axis=(1,2))
    error_bhp_max = error_bhp.max(axis=(1,2))

    error_oil_min = error_oil.min(axis=(1))
    error_oil_mean = error_oil.mean(axis=(1))
    error_oil_max = error_oil.max(axis=(1))

    os.makedirs(result_folder + '/imgs', exist_ok=True)

    show_error(poly_test.times, error_prod_min, error_prod_mean, error_prod_max, 'prod', result_folder + f'/imgs/error_prod.png')
    show_error(poly_test.times, error_bhp_min, error_bhp_mean, error_bhp_max, 'bhp', result_folder + f'/imgs/error_bhp.png')
    show_error(poly_test.times, error_oil_min, error_oil_mean, error_oil_max, 'prod_oil', result_folder + '/imgs/error_oil.png')

    show_hist_error(error_prod, error_bhp, error_oil, 'prod', 'bhp', 'prod_oil', 100, result_folder + '/imgs/hist.png')
    show.save_images(test_data_file, result_folder, delimiter)

if __name__ == '__main__':
    params_json_file = sys.argv[1]
    run(params_json_file)