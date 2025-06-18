import src.inno_reservoir.model.MultiModel as mm

import torch as tc
import numpy as np
import src.inno_reservoir.model.multipolygon as pg
import src.inno_reservoir.model.proxymodelcurve as mb

import src.inno_reservoir.model.ModelInterface as mi
import src.inno_reservoir.model.curvecontroller as cc

class MultiModelCurve(mm.MultiModel):
    def __init__(self, base: mb.ProxyModelNetCurve, poly: pg.MultPolygon, device = tc.device('cpu')):
        super().__init__(base, poly, device)

    def set_controller(self, controller: cc.IProxyController):
        self.base.set_controller(controller)

    def view_dim(self):
        return self.poly.well_count * 3

    def view(self, time_step: int, hidden_cur: tc.Tensor):
        bhp_bnd = self.poly.bhp_bnd[time_step]
        bhp_bnd = tc.from_numpy(bhp_bnd).type(tc.float32)

        status = self.poly.status[time_step]
        status = tc.from_numpy(status).type(tc.int8)

        wf = self.poly.wf_hist[time_step]
        wf = tc.from_numpy(wf).type(tc.float32)

        param = tc.cat((bhp_bnd, status, wf), dim=1).to(device=self.device)

        return self.base.view(hidden_cur, param)
    
    def veiw_weights(self, time_step: int, hidden_cur: tc.Tensor) -> tc.Tensor:
        bhp_bnd = self.poly.bhp_bnd[time_step]
        bhp_bnd = tc.from_numpy(bhp_bnd).type(tc.float32)

        status = self.poly.status[time_step]
        status = tc.from_numpy(status).type(tc.int8)

        wf = self.poly.wf_hist[time_step]
        wf = tc.from_numpy(wf).type(tc.float32)

        param = tc.cat((bhp_bnd, status, wf), dim=1).to(device=self.device)

        return self.base.veiw_weights(hidden_cur, param)
    
    def true_view(self, time_step: int):
        bhp = self.poly.bhp_hist[time_step]
        bhp = tc.from_numpy(bhp).type(tc.float32)

        prod = self.poly.prod_hist[time_step]
        prod = tc.from_numpy(prod).type(tc.float32)

        wf = self.poly.wf_hist[time_step]
        wf = tc.from_numpy(wf).type(tc.float32)

        sat = self.poly.sat_hist[time_step]
        sat = tc.from_numpy(sat).type(tc.float32)
        prod_w = prod * sat

        return tc.cat((bhp, prod - prod_w, prod_w), dim=1).to(device=self.device)
    
    def step(self, time_step: int, hidden_cur: tc.Tensor):
        dt = (self.poly.times[time_step + 1] - self.poly.times[time_step])
        self.base.dt.data = tc.tensor(dt, dtype=tc.float32, device=self.device)

        status = self.poly.status[time_step + 1]
        status = tc.from_numpy(status).type(tc.int8)

        wf = self.poly.wf_hist[time_step + 1]
        wf = tc.from_numpy(wf).type(tc.float32)

        param = tc.cat((status, wf), dim=1).to(device=self.device)

        hidden_next, bhp_bnd = self.base.step(hidden_cur, param)
        self.poly.bhp_bnd[time_step + 1] = bhp_bnd.cpu().numpy()

        return hidden_next
    
    def save_view(self, time_step: int, view: tc.Tensor):
        well_count = self.poly.well_count
        bhp, prod_o, prod_w = view.split(well_count, dim=1)

        bhp = bhp.cpu().numpy()
        prod_w = prod_w.cpu().numpy()
        prod_o = prod_o.cpu().numpy()
        prod = prod_o + prod_w

        sat_w = np.where(prod != 0.0, prod_w / prod, 0.0)

        self.poly.bhp_hist[time_step] = bhp
        self.poly.prod_hist[time_step] = prod
        self.poly.sat_hist[time_step] = sat_w
        