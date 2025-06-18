import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import os
from pathlib import Path

data_bhp = []
data_prod_w = []
data_prod_o = []
data_prod_g = []
data_wf = []

times = []

def get_well_data(df):
    columns = [col for col in df.columns if re.fullmatch('\d+', col)]

    data = []
    for col in columns:
        vals = np.array(df.loc[:, col].values, dtype=np.float64)
        data.append(vals)

    data = np.array(data).T
    return data

def get_peresure(pressure_df):
    pressures = []
    
    for well, well_df in pressure_df.groupby('well'):
        well_df = well_df.groupby('date')['value'].mean()
        pressure = np.array(well_df.values)
        pressures.append(pressure)

    pressures = np.array(pressures).T
    return pressures

for path in Path('data/source/variants').iterdir():
    if path.is_dir() and re.fullmatch('.*GDM_\d+', str(path)):
        gdm_name = path.name
        bhp = pd.read_csv(path.joinpath('wbhp.csv'), delimiter=';')

        wwpr = pd.read_csv(path.joinpath('wwpr.csv'), delimiter=';')
        wopr = pd.read_csv(path.joinpath('wopr.csv'), delimiter=';')
        wgpr = pd.read_csv(path.joinpath('wgpr.csv'), delimiter=';')
        wwir = pd.read_csv(path.joinpath('wwir.csv'), delimiter=';')

        rs = pd.read_csv(path.joinpath('rs.csv'), delimiter=';')
        wefac = pd.read_csv(path.joinpath('wefac.csv'), delimiter=';')

        date = pd.to_datetime(bhp['date'].values, format='%d.%m.%Y')
        times = (date - date[0]).days.to_numpy()

        #gdm_df = pd.read_csv(Path('gdm\multi_data\grids').joinpath(gdm_name + '.csv'), delimiter=';')
        #pressure_df = gdm_df[gdm_df.cube == 'PRESSURE']
        #peresure = get_peresure(pressure_df)        

        bhp = get_well_data(bhp)
        wwpr = get_well_data(wwpr)
        wopr = get_well_data(wopr)
        wgpr = get_well_data(wgpr)
        wwir = get_well_data(wwir)

        bhp = np.where(bhp < 1e-6, np.nan, bhp)
        #bhp = np.where(bhp < 1e-6, peresure, bhp)

        rs = get_well_data(rs)
        wefac = get_well_data(wefac)

        #wgpr /= rs + 1e-12
        wwpr = np.where(wwir > 0.0, -wwir, wwpr)
        wopr = np.where(wwir > 0.0, 0.0, wopr)
        wgpr = np.where(wwir > 0.0, 0.0, wgpr)

        #prod = wwpr + wopr + wgpr
        #sat = np.where(np.abs(prod) > 1e-7, wwpr / prod, 0.0)

        data_bhp.append(bhp)
        data_prod_w.append(wwpr)
        data_prod_o.append(wopr)
        data_prod_g.append(wgpr)
        data_wf.append(wefac)

        print(path)

data_bhp = np.transpose(np.array(data_bhp), (1, 0, 2))
data_prod_w = np.transpose(np.array(data_prod_w), (1, 0, 2))
data_prod_o = np.transpose(np.array(data_prod_o), (1, 0, 2))
data_prod_g = np.transpose(np.array(data_prod_g), (1, 0, 2))
data_wf = np.transpose(np.array(data_wf), (1, 0, 2))

data_prod = data_prod_w + data_prod_o# + data_prod_g
data_sat = np.where(np.abs(data_prod) > 1e-7, data_prod_w / data_prod, 0.0)

def save_ds(file_name, times, data_bhp, data_prod_w, data_prod_o, data_prod_g, data_wf, data_prod, data_sat):
    dfs = []

    var_count = data_bhp.shape[1]
    well_count = data_bhp.shape[2]

    for var_id in range(var_count):
        for well_id in range(well_count):
            df = pd.DataFrame({
                'date': times,
                'well_id': well_id,
                'variant_id': var_id,
                'bhp': data_bhp[:,var_id, well_id],
                'prod_w': data_prod_w[:,var_id, well_id],
                'prod_o': data_prod_o[:,var_id, well_id],
                'prod_g': data_prod_g[:,var_id, well_id],
                'prod': data_prod[:,var_id, well_id],
                'sat': data_sat[:,var_id, well_id],
                'wf': data_wf[:,var_id, well_id],
            })

            dfs.append(df)

    dfs = pd.concat(dfs)
    dfs.to_csv(file_name, sep='\t')

var_count = data_bhp.shape[1]
split = int(np.floor(0.7 * var_count))

var_score = data_prod.mean(axis=(0, 2))
print(var_score)

indices = np.argsort(var_score)
#indices = np.arange(var_count)
#np.random.shuffle(indices)

os.makedirs(f'data/dataset', exist_ok=True)

save_ds('data/dataset/train.txt', times, data_bhp[:,:split,:], 
    data_prod_w[:,:split,:], data_prod_o[:,:split,:], data_prod_g[:,:split,:], 
    data_wf[:,:split,:], data_prod[:,:split,:], data_sat[:,:split,:])

save_ds('data/dataset/test.txt', times, data_bhp[:,split:,:], 
    data_prod_w[:,split:,:], data_prod_o[:,split:,:], data_prod_g[:,split:,:], 
    data_wf[:,split:,:], data_prod[:,split:,:], data_sat[:,split:,:])
