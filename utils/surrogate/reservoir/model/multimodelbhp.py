import model.MultiModel as mm

import torch as tc
import numpy as np
import model.multipolygon as pg
import model.model_base as mb

import model.ModelInterface as mi

class MultiModelBHP(mm.MultiModel):
    def __init__(self, base: mb.IProxyModel, poly: pg.MultPolygon, device = tc.device('cpu')):
        super().__init__(base, poly, device)

    def view(self, time_step: int, hidden_cur: tc.Tensor):
        bhp = self.poly.bhp_hist[time_step]
        bhp = tc.from_numpy(bhp).type(tc.float32)

        wf = self.poly.wf_hist[time_step]
        wf = tc.from_numpy(wf).type(tc.float32)

        param = tc.cat((bhp, wf), dim=1).to(device= self.device)

        return self.base.view(hidden_cur, param)
    
    def veiw_weights(self, time_step: int, hidden_cur: tc.Tensor) -> tc.Tensor:
        bhp = self.poly.bhp_hist[time_step]
        bhp = tc.from_numpy(bhp).type(tc.float32)

        prod = self.poly.prod_hist[time_step]
        prod = tc.from_numpy(prod).type(tc.float32)

        sat = self.poly.sat_hist[time_step]
        sat = tc.from_numpy(sat).type(tc.float32)

        prod_w = prod * sat

        wf = self.poly.wf_hist[time_step]
        wf = tc.from_numpy(wf).type(tc.float32)

        param = tc.cat((bhp, wf, prod - prod_w, prod_w), dim=1).to(device=self.device)

        return self.base.veiw_weights(hidden_cur, param)
    
    def true_view(self, time_step: int):
        prod = self.poly.prod_hist[time_step]
        prod = tc.from_numpy(prod).type(tc.float32)

        wf = self.poly.wf_hist[time_step]
        wf = tc.from_numpy(wf).type(tc.float32)

        sat = self.poly.sat_hist[time_step]
        sat = tc.from_numpy(sat).type(tc.float32)

        prod_w = prod * sat

        #prod_w = tc.where(wf < 1e-6, 0.0, prod_w)
        #prod = tc.where(wf < 1e-6, 0.0, prod)

        return tc.cat((prod - prod_w, prod_w), dim=1).to(device= self.device)
    
    def step(self, time_step: int, hidden_cur: tc.Tensor):
        dt = (self.poly.times[time_step + 1] - self.poly.times[time_step])
        self.base.dt.data = tc.tensor(dt, dtype=tc.float32, device=self.device)

        bhp = self.poly.bhp_hist[time_step + 1]
        bhp = tc.from_numpy(bhp).type(tc.float32)

        wf = self.poly.wf_hist[time_step + 1]
        wf = tc.from_numpy(wf).type(tc.float32)

        param = tc.cat((bhp, wf), dim=1).to(device=self.device)

        #batch_size = hidden_cur.shape[0]
        #param = param.expand((batch_size, -1))

        hidden_next = self.base.step(hidden_cur, param)
        return hidden_next
    
    def save_view(self, time_step: int, view: tc.Tensor):
        well_count = self.poly.well_count
        prod_o, prod_w = view.split(well_count, dim=1)

        prod_w = prod_w.cpu().numpy()
        prod_o = prod_o.cpu().numpy()
        prod = prod_o + prod_w

        sat_w = np.where(prod != 0.0, prod_w / prod, 0.0)

        self.poly.prod_hist[time_step] = prod
        self.poly.sat_hist[time_step] = sat_w