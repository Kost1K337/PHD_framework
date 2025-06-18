import numpy as np
from io import StringIO
import pandas as pd

class Polygon:
    well_count: int
    bhp_hist: np.ndarray
    prod_hist: np.ndarray
    sat_hist: np.ndarray
    wf_hist: np.ndarray

    times: np.ndarray

    pressure: np.ndarray
    saturation: np.ndarray
    pressure_cor: np.ndarray
    saturation_cor: np.ndarray

    def __init__(self) :
        pass

    def read_history(self, times_name: str, bhp_name: str, prod_name: str, sat_name: str, wf_name: str = None):
        self.times = np.loadtxt(times_name)
        self.bhp_hist = np.loadtxt(bhp_name)
        self.prod_hist = np.loadtxt(prod_name)
        self.sat_hist = np.loadtxt(sat_name)

        if wf_name is None:
            self.wf_hist = np.ones_like(self.prod_hist)
        else:
            self.wf_hist = np.loadtxt(wf_name)

        self.well_count = self.prod_hist.shape[1]
        time_steps_count = self.times.shape[0]

        self.pressure = np.zeros((time_steps_count, self.well_count))
        self.saturation = np.zeros((time_steps_count, self.well_count))

        self.pressure_cor = np.zeros((time_steps_count, self.well_count))
        self.saturation_cor = np.zeros((time_steps_count, self.well_count))

    def read_history_csv(self, data_file_name, delimiter='\t'):
        df = pd.read_csv(data_file_name, delimiter=delimiter)

        names = [
            'bhp',
            'prod_w',
            'prod_o',
            'prod_g',
            'prod',
            'sat',
            'wf',
        ]

        data = {}
        for name in names:
            data[name] = []

        for well_id, well_df in df.groupby('well_id'):
            for name in names:
                vals = np.array(well_df[name].values)
                data[name].append(vals)
            
            self.times = np.array(well_df['date'].values)

        for name in names:
            data[name] = np.array(data[name])

        for name in names:
            vals = np.array(data[name])
            vals = np.transpose(vals, (1, 0))
            data[name] = vals

        self.bhp_hist = data['bhp']
        self.prod_hist = data['prod']
        self.sat_hist = data['sat']
        self.wf_hist = data['wf']

        self.well_count = self.prod_hist.shape[1]
        time_steps_count = self.times.shape[0]

        self.bhp_hist = self.bhp_hist.reshape((time_steps_count, self.well_count))
        self.prod_hist = self.prod_hist.reshape((time_steps_count, self.well_count))
        self.sat_hist = self.sat_hist.reshape((time_steps_count, self.well_count))
        self.wf_hist = self.wf_hist.reshape((time_steps_count, self.well_count))

        self.pressure = np.zeros((time_steps_count, self.well_count))
        self.saturation = np.zeros((time_steps_count, self.well_count))

        self.pressure_cor = np.zeros((time_steps_count, self.well_count))
        self.saturation_cor = np.zeros((time_steps_count, self.well_count))


    def print_results(self, bhp_name: str, sat_well_name: str, pres_name: str, sat_name: str):
        np.savetxt(bhp_name, self.bhp_hist, delimiter='\t')
        np.savetxt(sat_well_name, self.sat_hist, delimiter='\t')

        np.savetxt(pres_name, self.pressure, delimiter='\t')
        np.savetxt(sat_name, self.saturation, delimiter='\t')

    def normalize(self):
        mean_bhp = self.bhp_hist.mean()
        disp_bhp = self.bhp_hist.std()

        prod = self.prod_hist[np.abs(self.prod_hist) > 1e-7]

        mean_prod = 0.0 # self.prod_hist.mean()
        disp_prod = np.sqrt((prod * prod).mean() - mean_prod * mean_prod)

        offset_bhp = -mean_bhp
        mult_bhp = 1.0 / disp_bhp

        print(mean_bhp, disp_bhp)

        offset_prod = -mean_prod
        mult_prod = 1.0 / (disp_prod)

        self.bhp_hist = (self.bhp_hist + offset_bhp) * mult_bhp
        self.prod_hist = (self.prod_hist + offset_prod) * mult_prod


    def copy(self):
        poly = Polygon()

        poly.well_count = self.well_count

        poly.bhp_hist = self.bhp_hist.copy()
        poly.prod_hist = self.prod_hist.copy()
        poly.sat_hist = self.sat_hist.copy()

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

        self.times = poly.times.copy()

        self.pressure = poly.pressure.copy()
        self.saturation = poly.saturation.copy()
        self.pressure_cor = poly.pressure_cor.copy()
        self.saturation_cor = poly.saturation_cor.copy()

