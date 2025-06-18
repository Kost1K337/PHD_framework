import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

df = pd.read_csv('data/source/data_06_06.csv', delimiter=';')
df_wwir = pd.read_csv('data/source/wwir.csv', delimiter=';',dtype=str)
df_wefac = pd.read_csv('data/source/wefacs.csv', delimiter=';',dtype=str)

datas = {}

def calc_pressure(well, bhp, prod):
    def to_matrix(df):
        columns = [col for col in df.columns if re.fullmatch('\(\d+, \d+, \d+\)', col)]

        data = []
        for col in columns:
            vals = np.array(df.loc[:, col].values, dtype=np.float64)
            data.append(vals)

        data = np.array(data).T
        return data

    pressure = pd.read_csv(f'data/source/dataset_gdm2_pressure/{well}_0.csv')
    krw = pd.read_csv(f'data/source/dataset_gdm2_krw/{well}_0.csv')
    kro = pd.read_csv(f'data/source/dataset_gdm2_kro/{well}_0.csv')

    pressure = to_matrix(pressure)
    krw = to_matrix(krw)
    kro = to_matrix(kro)

    krt = krw + kro

    pressure_orig = pressure.copy()
    krt_orig = krt.copy()
    #prod_orig = prod.copy()

    bhp_mask = (bhp > 0.0)
    pressure = pressure[bhp_mask, :]
    krt = krt[bhp_mask, :]
    prod = prod[bhp_mask]
    bhp = bhp[bhp_mask]    

    cell_count = pressure.shape[1]

    f1 = krt * (pressure - bhp[:, None])
    f2 = krt
    X = np.concatenate((f1, f2), axis=1)

    A = X.T @ X
    R = X.T @ prod
    A_inv = np.linalg.pinv(A)

    w = A_inv @ R
    #w = np.linalg.solve(A + np.eye(cell_count * 2) * 1e-12, R)

    J = w[:cell_count]
    rJ = w[cell_count:]

    sJ = krt_orig @ J

    q0 = ((pressure_orig * krt_orig) @ J) + (krt_orig @ rJ)
    p0 = q0 / sJ

    return p0



for model, df_model in df.groupby('Model'):
    for sc, group in df_model.groupby('Scenario'):
        data_bhp = []
        data_prod_w = []
        data_prod_o = []
        data_prod_g = []
        data_wf = []

        for well, well_df in group.groupby('Well'):
            bhp = np.array(well_df['WBHP'].values)
            pressure0 = np.array(well_df['Pressure'].values)

            WWPR = np.array(well_df['WWPR'].values)
            WOPR = np.array(well_df['WOPR'].values)
            WGPR = np.array(well_df['WGPR'].values)
            Rs = np.array(well_df['Rs'].values)
            WGPR /= Rs

            WWIR = np.array(df_wwir.loc[:,'Well_'+str(well)].values, dtype=np.float32)
            WWPR = np.where(WWIR > 0.0, -WWIR, WWPR)
            WOPR = np.where(WWIR > 0.0, 0.0, WOPR)
            WGPR = np.where(WWIR > 0.0, 0.0, WGPR)

            WEFAC = np.array(df_wefac.loc[:,'Well_'+str(well)].values, dtype=np.float32)

            prod = WWPR + WOPR + WGPR

            if prod.dot(prod) < 1e-7:
                continue

            #pressure = calc_pressure(well, bhp, prod)
            #bhp = np.where(np.abs(bhp) > 0.0, bhp, pressure)

            bhp = np.where(np.abs(bhp) > 0.0, bhp, pressure0)

            data_bhp.append(bhp)
            data_prod_w.append(WWPR)
            data_prod_o.append(WOPR)
            data_prod_g.append(WGPR)
            data_wf.append(WEFAC)

            date = pd.to_datetime(well_df['Date'].values)
            times = (date - date[0]).days.to_numpy()

        data_bhp = np.array(data_bhp).T
        data_prod_w = np.array(data_prod_w).T
        data_prod_o = np.array(data_prod_o).T
        data_prod_g = np.array(data_prod_g).T
        data_wf = np.array(data_wf).T

        datas[(model, sc)] = (times, data_bhp, data_prod_w, data_prod_o, data_prod_g, data_wf)

well_count = 0

for key in datas:
    times, data_bhp, data_prod_w, data_prod_o, data_prod_g, data_wf = datas[key]

    data_prod = data_prod_w + data_prod_o + data_prod_g
    sat = np.where(np.abs(data_prod) > 1e-7, data_prod_w / data_prod, 0.0)

    datas[key] = (times, data_bhp, data_prod_w, data_prod_o, data_prod_g, data_prod, sat, data_wf)
    well_count = max(well_count, data_bhp.shape[1])

dfs = {}

for key, data in datas.items():
    times = data[0]
    date = np.repeat(times, well_count)
    index = np.arange(well_count * times.shape[0])
    well_index = index % well_count

    df = pd.DataFrame({
            'date': date,
            'well_id': well_index,
            'bhp':  data[1].flatten(),
            'prod_w': data[2].flatten(),
            'prod_o': data[3].flatten(),
            'prod_g': data[4].flatten(),
            'prod': data[5].flatten(),
            'sat': data[6].flatten(),
            'wf': data[7].flatten(),
        })

    dfs[key] = df

os.makedirs(f'data/dataset/orig', exist_ok=True)

for key, df in dfs.items():
    df.to_csv(f"data/dataset/orig/gdm_{key}.txt", sep='\t')
    print(f"saved: data/dataset/orig/gdm_{key}.txt")