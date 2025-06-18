import torch as tc
import numpy as np
import os
import sys
import json
import model.multipolygon as mpg
from model.proxymodelnet import ProxyModelNet
from model.proxymodelbhp import ProxyModelNetBHP
from model.proxymodelmix import ProxyModelNetMix
from model.MultiModel import MultiModel
from model.multimodelbhp import MultiModelBHP
from model.multimodelmix import MultiModelMix
import trainer.Optimazers as opt
import trainer.MultiMLTrainer as mmt

import show

from torch.utils.tensorboard import SummaryWriter

def init_disps_view(poly: mpg.MultPolygon, bhp_control: bool):
    well_count = poly.well_count
    disps_hidden = np.zeros(well_count * 2)
    disps_view = np.zeros(well_count * 2)

    prod_w = poly.prod_hist #* poly.sat_hist

    disp_bhp = np.nanstd(poly.bhp_hist, axis=(0, 1))
    disp_prod_w = np.sqrt((prod_w * prod_w).mean(axis=(0, 1)))

    disp_bhp = np.where(np.isnan(disp_bhp), np.nanmean(disp_bhp), disp_bhp)
    disp_bhp = np.where(disp_bhp < 1e-7, np.nanmean(disp_bhp), disp_bhp)
    disp_prod_w = np.where(disp_prod_w < 1e-7, disp_prod_w.mean(), disp_prod_w)

    disps_view[:well_count] = disp_prod_w.mean() if bhp_control else disp_bhp.mean()
    disps_view[well_count:] = disp_prod_w.mean()

    #disps_hidden[:well_count] = disp_bhp.mean()
    #disps_hidden[well_count:well_count*2] = 1.0
    disps_hidden[:] = 1.0

    return disps_hidden, disps_view

def run(params_json_file):
    with open(params_json_file) as params_file:
        params = json.load(params_file)

    print(params)
    print('---------------------------------------------------')

    train_data_file = params['train_data_file']
    test_data_file = params['test_data_file']
    delimiter = params['delimiter']

    result_folder = params['result_folder']
    model_filename = params['model_filename']
    epochs = params['epochs']
    stop_well_weight = params['stop_well_weight']
    bhp_control = params['bhp_control']

    hidden_per_well = params['hidden_per_well']
    addition_cells_mult = params['addition_cells_per_well']

    learning_rate_base = params['learning_rate_base']
    learning_rate_net = params['learning_rate_net']
    time_split_ratio = params['time_split_ratio']

    device = tc.device(params['device'])
    tc.manual_seed(0)

    print('train data:', train_data_file)
    print('test data:', test_data_file)
    print('---------------------------------------------------')

    poly = mpg.MultPolygon()
    poly.read_history(train_data_file, delimiter=delimiter)

    poly_test = mpg.MultPolygon()
    poly_test.read_history(test_data_file, delimiter=delimiter)

    well_count = poly.well_count
    addition_cells = int(np.round(well_count * addition_cells_mult))

    if bhp_control:
        model = ProxyModelNetBHP(well_count, 1.0, hidden_per_well, addition_cells)
    else:
        model = ProxyModelNet(well_count, 1.0, hidden_per_well, addition_cells)

    model.stop_well_weight = stop_well_weight   

    os.makedirs(result_folder, exist_ok=True)

    if os.path.exists(model_filename):
        model.load_state_dict(tc.load(model_filename))
        offset_bhp, mult_bhp, offset_prod, mult_prod = model.get_scales()

        poly.scale(offset_bhp, mult_bhp, offset_prod, mult_prod)
        poly_test.scale(offset_bhp, mult_bhp, offset_prod, mult_prod)
    else:
        offset_bhp, mult_bhp, offset_prod, mult_prod = poly.normalize()
        poly_test.scale(offset_bhp, mult_bhp, offset_prod, mult_prod) 

        model.init_params(poly.prod_hist, poly.sat_hist, poly.bhp_hist, poly.times)
        model.save_scales(offset_bhp, mult_bhp, offset_prod, mult_prod)

    model = model.to(device=device)

    smodel = MultiModelBHP(model, poly, device=device) if bhp_control else MultiModel(model, poly, device=device)
    smodel_test = MultiModelBHP(model, poly_test, device=device) if bhp_control else MultiModel(model, poly_test, device=device)
    optim = opt.ProxyOptimazerNet(model, learning_rate_base, learning_rate_net)

    trainer = mmt.MultiTrainer(smodel, smodel_test, optim, device=device)

    disps_hidden, disps_view = init_disps_view(poly, bhp_control)

    print("disps_view: ", disps_view)
    print("disps_view_min:", disps_view.min())

    trainer.set_disps_hidden(disps_hidden)
    trainer.set_disps_view(disps_view)    

    if time_split_ratio < 1.0:
        split = int(np.floor(poly.times.shape[0] * time_split_ratio))
        well_ids = np.where(np.abs(poly.prod_hist[:split]).mean(axis=(0,1)) > 0.0)
        print('disabled wells: ', well_ids)

        if bhp_control:
            poly.prod_hist[split:,:,well_ids] = np.nan
            poly_test.prod_hist[:split,:,well_ids] = np.nan
        else:
            poly.bhp_hist[split:,:,well_ids] = np.nan   
            poly_test.bhp_hist[:split,:,well_ids] = np.nan

        poly.sat_hist[split:,:,well_ids] = np.nan
        poly_test.sat_hist[:split,:,well_ids] = np.nan

    min_loss_base = 1e6
    min_loss_test = 1e6
    min_loss_train = 1e6
    iter = 0

    best_base_weights = model.save_base_params()

    writer = SummaryWriter()

    while min_loss_test > 1e-6:
        if iter >= epochs:
            break
        
        model.setNetEnabled(False)
        loss_base = trainer.epoch_ml()

        if loss_base < min_loss_base:
            min_loss_base = loss_base
            loss_train = loss_base
            best_base_weights = model.save_base_params()
            useNet = False
        else:
            model.setNetEnabled(True)
            base_params = model.save_base_params()
            model.load_base_params(best_base_weights)
            loss_train = trainer.epoch_ml()
            model.load_base_params(base_params)
            useNet = True

        loss_test = trainer.epoch_test()

        if time_split_ratio < 1.0:
            loss_train = loss_train / time_split_ratio
            loss_test = loss_test / (1 - time_split_ratio)

        if np.isnan(loss_train) or np.isnan(loss_test):
            print(iter, loss_base, min_loss_test, loss_train, loss_test)
            break

        if loss_train < min_loss_train:
            min_loss_train = loss_train

        if loss_test < min_loss_test:
            min_loss_test = loss_test

            poly_copy = poly_test.copy()
            trainer.make_history()
            poly_test.unscale(offset_bhp, mult_bhp, offset_prod, mult_prod)
            poly_test.print_results(result_folder + "/result.feth")

            model.save(result_folder + "/model.txt")
            tc.save(model.state_dict(), result_folder + "/model.tc")
            poly_test.assign(poly_copy)

        print(iter, min_loss_base, min_loss_train, min_loss_test, loss_base, loss_train, loss_test, useNet)

        writer.add_scalars('Losses', {
            'base': loss_base,
            'train': loss_train,
            'test': loss_test,
            }, iter)

        writer.add_scalars('Min losses', {
            'min train': min_loss_train,
            'min test': min_loss_test,
            }, iter)
        
        iter += 1


if __name__ == '__main__':
    params_json_file = sys.argv[1]
    run(params_json_file)