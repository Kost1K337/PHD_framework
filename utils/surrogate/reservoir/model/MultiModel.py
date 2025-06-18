import torch as tc
import torch.nn as nn
import numpy as np
import src.inno_reservoir.model.multipolygon as pg
import src.inno_reservoir.model.model_base as mb

import src.inno_reservoir.model.ModelInterface as mi

class MultiModel(mi.SModelInterface):
    base: mb.IProxyModel

    def __init__(self, base: mb.IProxyModel, poly: pg.MultPolygon, device = tc.device('cpu')):
        self.base = base
        self.poly = poly
        self.device = device

    def hidden_dim(self):
        return self.base.hidden_dim()

    def view_dim(self):
        return self.poly.well_count * 2

    def time_steps(self):
        return self.poly.times.shape[0]

    def start(self):
        hidden = self.base.start()
        var_count = self.poly.bhp_hist.shape[1]
        hidden = hidden.expand((var_count, -1))
        return hidden

    def view(self, time_step: int, hidden_cur: tc.Tensor):
        prod = self.poly.prod_hist[time_step]
        prod = tc.from_numpy(prod).type(tc.float32)

        wf = self.poly.wf_hist[time_step]
        wf = tc.from_numpy(wf).type(tc.float32)

        param = tc.cat((prod, wf), dim=1).to(device= self.device)

        return self.base.view(hidden_cur, param)
    
    def veiw_weights(self, time_step: int, hidden_cur: tc.Tensor) -> tc.Tensor:
        prod = self.poly.prod_hist[time_step]
        prod = tc.from_numpy(prod).type(tc.float32)

        wf = self.poly.wf_hist[time_step]
        wf = tc.from_numpy(wf).type(tc.float32)

        param = tc.cat((prod, wf), dim=1).to(device= self.device)

        return self.base.veiw_weights(hidden_cur, param)

    def true_view(self, time_step: int):
        bhp = self.poly.bhp_hist[time_step]
        bhp = tc.from_numpy(bhp).type(tc.float32)

        prod = self.poly.prod_hist[time_step]
        prod = tc.from_numpy(prod).type(tc.float32)

        sat = self.poly.sat_hist[time_step]
        sat = tc.from_numpy(sat).type(tc.float32)

        prod_w = prod * sat

        return tc.cat((bhp, prod_w), dim=1).to(device= self.device)

    def step(self, time_step: int, hidden_cur: tc.Tensor):
        dt = (self.poly.times[time_step + 1] - self.poly.times[time_step])
        self.base.dt.data = tc.tensor(dt, dtype=tc.float32, device=self.device)

        prod = self.poly.prod_hist[time_step + 1]
        prod = tc.from_numpy(prod).type(tc.float32)

        wf = self.poly.wf_hist[time_step + 1]
        wf = tc.from_numpy(wf).type(tc.float32)

        param = tc.cat((prod, wf), dim=1).to(device=self.device)

        #batch_size = hidden_cur.shape[0]
        #param = param.expand((batch_size, -1))

        hidden_next = self.base.step(hidden_cur, param)
        return hidden_next

    def save_hidden(self, time_step: int, hidden: tc.Tensor):
        cell_count = hidden.shape[1] // 2
        well_count = self.poly.well_count
        pressure, x_saturation = hidden.split(cell_count, dim=1)

        pressure = pressure[:, :well_count]
        x_saturation = x_saturation[:, :well_count]

        self.poly.pressure[time_step] = pressure.detach().cpu().numpy()
        self.poly.saturation[time_step] = x_saturation.detach().cpu().numpy()

    def save_hidden_cor(self, time_step: int, hidden: tc.Tensor):
        cell_count = hidden.shape[1] // 2
        well_count = self.poly.well_count
        pressure, x_saturation = hidden.split(cell_count, dim=1)

        pressure = pressure[:, :well_count]
        x_saturation = x_saturation[:, :well_count]

        self.poly.pressure_cor[time_step] = pressure.detach().cpu().numpy()
        self.poly.saturation_cor[time_step] = x_saturation.detach().cpu().numpy()

    def save_view(self, time_step: int, view: tc.Tensor):
        well_count = self.poly.well_count
        bhp, prod_w = view.split(well_count, dim=1)

        prod_w = prod_w.cpu().numpy()
        prod = self.poly.prod_hist[time_step]

        sat_w = np.where(prod != 0.0, prod_w / prod, 0.0)

        self.poly.bhp_hist[time_step] = bhp.cpu().numpy()
        self.poly.sat_hist[time_step] = sat_w
