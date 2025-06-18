import pandas as pd
import numpy as np
from io import StringIO
import src.inno_reservoir.model.polygon as pg

class MultPolygon:
    well_count: int
    var_count: int
    bhp_hist: np.ndarray
    prod_hist: np.ndarray
    sat_hist: np.ndarray
    wf_hist: np.ndarray

    status: np.ndarray
    bhp_bnd: np.ndarray
    prod_bnd: np.ndarray

    times: np.ndarray

    pressure: np.ndarray
    saturation: np.ndarray
    pressure_cor: np.ndarray
    saturation_cor: np.ndarray

    def __init__(self) :
        pass

    def read_history(self, data_file_name, delimiter='\t'):
        df = pd.read_csv(data_file_name, delimiter=delimiter)

        names = [
            'bhp',
            'prod',
            'sat',
            'wf',
            'status',
            'bhp_bnd',
            'prod_bnd',
        ]

        data = {}
        not_exist_columns = set()

        for name in names:
            data[name] = []

        for var_id, var_df in df.groupby('variant_id'):
            var_data = {}
            
            for name in names:
                var_data[name] = []

            for well_id, well_df in var_df.groupby('well_id'):
                for name in names:
                    if name in well_df.columns:
                        vals = np.array(well_df[name].values)
                        var_data[name].append(vals)
                    else:
                        not_exist_columns.add(name)
                
                self.times = np.array(well_df['date'].values)

            for name in names:
                if name not in not_exist_columns:
                    var_data[name] = np.array(var_data[name], dtype=np.float64)
                    data[name].append(var_data[name])

        for name in names:
            if name not in not_exist_columns:
                vals = np.array(data[name])
                vals = np.transpose(vals, (2, 0, 1))
                data[name] = vals

        self.bhp_hist = data['bhp']
        self.prod_hist = data['prod']
        # self.prod_hist = data['prod_w']*1022 + data['prod_o']*831.519184929611 + data['prod_g']*1.09619179967901
        self.sat_hist = data['sat']
        # self.sat_hist[self.prod_hist>0] = 1 - (data['prod_w'][self.prod_hist>0]*1022)/self.prod_hist[self.prod_hist>0]
        self.wf_hist = data['wf']

        self.wf_hist = np.where(np.abs(self.prod_hist) < 1e-6, 0.0, self.wf_hist)

        if 'status' in not_exist_columns:
            self.status = np.where(self.prod_hist > 0.0, 1, -1)
            self.status = np.where(np.abs(self.wf_hist) < 1e-6, 0, self.status)
        else:
            self.status = data['status']

        if 'bhp_bnd' in not_exist_columns:
            self.bhp_bnd = self.bhp_hist.copy()
        else:
            self.bhp_bnd = data['bhp_bnd']

        if 'prod_bnd' in not_exist_columns:
            self.prod_bnd = self.prod_hist
            #self.prod_bnd = self.status * 1e9
        else:
            self.prod_bnd = data['prod_bnd']

        self.well_count = self.prod_hist.shape[2]
        time_steps_count = self.times.shape[0]
        self.var_count = self.prod_hist.shape[1]

        self.bhp_hist = self.bhp_hist.reshape((time_steps_count, self.var_count, self.well_count))
        self.prod_hist = self.prod_hist.reshape((time_steps_count, self.var_count, self.well_count))
        self.sat_hist = self.sat_hist.reshape((time_steps_count, self.var_count, self.well_count))
        self.wf_hist = self.wf_hist.reshape((time_steps_count, self.var_count, self.well_count))

        self.status = self.status.reshape((time_steps_count, self.var_count, self.well_count))
        self.bhp_bnd = self.bhp_bnd.reshape((time_steps_count, self.var_count, self.well_count))
        self.prod_bnd = self.prod_bnd.reshape((time_steps_count, self.var_count, self.well_count))

        self.pressure = np.zeros((time_steps_count, self.var_count, self.well_count))
        self.saturation = np.zeros((time_steps_count, self.var_count, self.well_count))

        self.pressure_cor = np.zeros((time_steps_count, self.var_count, self.well_count))
        self.saturation_cor = np.zeros((time_steps_count, self.var_count, self.well_count))        

    def from_polygon(self, poly):
        self.var_count = 1
        self.well_count = poly.well_count

        self.times = poly.times

        self.bhp_hist = poly.bhp_hist[:, None, :]
        self.prod_hist = poly.prod_hist[:, None, :]
        self.sat_hist = poly.sat_hist[:, None, :]
        self.wf_hist = poly.wf_hist[:, None, :]

        self.pressure = poly.pressure[:, None, :]
        self.saturation = poly.saturation[:, None, :]

        self.pressure_cor = poly.pressure_cor[:, None, :]
        self.saturation_cor = poly.saturation_cor[:, None, :]

        self.status = np.where(self.prod_hist > 0.0, 1, -1)
        self.status = np.where(np.abs(self.wf_hist) < 1e-6, 0, self.status)
        self.bhp_bnd = self.bhp_hist.copy()
        self.prod_bnd = self.status * 1e9

    def add_polygon(self, poly: pg.Polygon):
        self.var_count += 1

        self.bhp_hist = np.concatenate((self.bhp_hist, poly.bhp_hist[:, None, :]), axis=1)
        self.prod_hist = np.concatenate((self.prod_hist, poly.prod_hist[:, None, :]), axis=1)
        self.sat_hist = np.concatenate((self.sat_hist, poly.sat_hist[:, None, :]), axis=1)
        self.wf_hist = np.concatenate((self.wf_hist, poly.wf_hist[:, None, :]), axis=1)

        self.pressure = np.concatenate((self.pressure, poly.pressure[:, None, :]), axis=1)
        self.saturation = np.concatenate((self.saturation, poly.saturation[:, None, :]), axis=1)

        self.pressure_cor = np.concatenate((self.pressure_cor, poly.pressure_cor[:, None, :]), axis=1)
        self.saturation_cor = np.concatenate((self.saturation_cor, poly.saturation_cor[:, None, :]), axis=1)

        status = np.where(poly.prod_hist > 0.0, 1, -1)
        status = np.where(np.abs(poly.wf_hist) < 1e-6, 0, status)
        bhp_bnd = poly.bhp_hist.copy()
        prod_bnd = status * 1e9

        self.status = np.concatenate((self.status, status[:, None, :]), axis=1)
        self.bhp_bnd = np.concatenate((self.bhp_bnd, bhp_bnd[:, None, :]), axis=1)
        self.prod_bnd = np.concatenate((self.prod_bnd, prod_bnd[:, None, :]), axis=1)


    def print_results(self, file_name: str):
        #dfs = []

        var_count = self.var_count
        well_count = self.well_count

        date = np.repeat(self.times, well_count * var_count)

        index = np.arange(well_count * var_count * self.times.shape[0])
        well_index = index % well_count
        var_index = (index // well_count) % var_count

        df = pd.DataFrame({
                    'date': date,
                    'well_id': well_index,
                    'variant_id': var_index,
                    'bhp': self.bhp_hist.flatten(),
                    'prod': self.prod_hist.flatten(),
                    'sat': self.sat_hist.flatten(),
                    'pressure': self.pressure.flatten(),
                    'saturation': self.saturation.flatten(),
                    'wf': self.wf_hist.flatten(),
                })

        df.to_feather(file_name)
        
    def normalize(self):
        mean_bhp = np.nanmean(self.bhp_hist)
        disp_bhp = np.nanstd(self.bhp_hist)

        prod = self.prod_hist[np.abs(self.prod_hist) > 1e-7]

        mean_prod = 0.0 # self.prod_hist.mean()
        disp_prod = np.sqrt((prod * prod).mean() - mean_prod * mean_prod)

        offset_bhp = -mean_bhp
        mult_bhp = 1.0 / disp_bhp

        print(mean_bhp, disp_bhp)

        offset_prod = -mean_prod
        mult_prod = 1.0 / (disp_prod)

        self.scale(offset_bhp, mult_bhp, offset_prod, mult_prod)
        return offset_bhp, mult_bhp, offset_prod, mult_prod

    def scale(self, offset_bhp, mult_bhp, offset_prod, mult_prod):
        self.bhp_hist = (self.bhp_hist + offset_bhp) * mult_bhp
        self.prod_hist = (self.prod_hist + offset_prod) * mult_prod

        self.bhp_bnd = (self.bhp_bnd + offset_bhp) * mult_bhp
        self.prod_bnd = (self.prod_bnd + offset_prod) * mult_prod

    def unscale(self, offset_bhp, mult_bhp, offset_prod, mult_prod):
        self.bhp_hist = self.bhp_hist / mult_bhp - offset_bhp
        self.pressure = self.pressure / mult_bhp - offset_bhp
        self.prod_hist = self.prod_hist / mult_prod - offset_prod

        self.bhp_bnd = self.bhp_bnd / mult_bhp - offset_bhp
        self.prod_bnd = self.prod_bnd / mult_bhp - offset_bhp

    def copy(self):
        poly = MultPolygon()

        poly.well_count = self.well_count
        poly.var_count = self.var_count

        poly.bhp_hist = self.bhp_hist.copy()
        poly.prod_hist = self.prod_hist.copy()
        poly.sat_hist = self.sat_hist.copy()
        poly.wf_hist = self.wf_hist.copy()

        poly.status = self.status.copy()
        poly.bhp_bnd = self.bhp_bnd.copy()
        poly.prod_bnd = self.prod_bnd.copy()

        poly.times = self.times.copy()

        poly.pressure = self.pressure.copy()
        poly.saturation = self.saturation.copy()
        poly.pressure_cor = self.pressure_cor.copy()
        poly.saturation_cor = self.saturation_cor.copy()

        return poly

    def assign(self, poly):
        self.well_count = poly.well_count

        self.bhp_hist = poly.bhp_hist.copy()
        self.prod_hist = poly.prod_hist.copy()
        self.sat_hist = poly.sat_hist.copy()
        self.wf_hist = poly.wf_hist.copy()

        self.status = poly.status.copy()
        self.bhp_bnd = poly.bhp_bnd.copy()
        self.prod_bnd = poly.prod_bnd.copy()

        self.times = poly.times.copy()

        self.pressure = poly.pressure.copy()
        self.saturation = poly.saturation.copy()
        self.pressure_cor = poly.pressure_cor.copy()
        self.saturation_cor = poly.saturation_cor.copy()


