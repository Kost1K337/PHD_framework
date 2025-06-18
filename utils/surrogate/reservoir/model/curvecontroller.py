from src.mipt_solver.nodal.inflow_curves import Curve
from src.inno_reservoir.curves.iprcurve import IPRCurve, WATCurve, OILCurve, GASCurve
import torch as tc
import numpy as np

from typing import Tuple, List

class IProxyControllerCurve:
    def control(self, curves: np.ndarray) -> np.ndarray:
        pass

class IProxyController:
    def control(self, PI0: tc.Tensor, pressure0: tc.Tensor, rmob: tc.Tensor, sat0: tc.Tensor) -> tc.Tensor:
        pass

class ProxyController(IProxyController):
    def __init__(self, controller: IProxyControllerCurve):
        self.controller = controller

    def control(self, PI0: tc.Tensor, pressure0: tc.Tensor, rmob: tc.Tensor, sat0: tc.Tensor, ds: tc.Tensor) -> tc.Tensor:
        # PI0 -PI для всего фонда, 
        device = PI0.device
        PI0 = PI0.cpu().numpy()
        pressure0 = pressure0.cpu().numpy()
        rmob = rmob.cpu().numpy()
        sat0 = sat0.cpu().numpy()
        ds = ds.cpu().numpy()

        curves = np.empty(PI0.shape, dtype=object)
        if len(rmob.shape) == 2:
            for var in range(curves.shape[0]):
                for well in range(curves.shape[1]):
                    cur_curves = []
                    cur_curves.append(IPRCurve(PI0[var,well], pressure0[var,well], rmob[var,well], sat0[var,well], ds[var,well]))
                    cur_curves.append(WATCurve(PI0[var,well], pressure0[var,well], rmob[var,well], sat0[var,well], ds[var,well]))
                    cur_curves.append(OILCurve(PI0[var,well], pressure0[var,well], rmob[var,well], sat0[var,well], ds[var,well]))
                    cur_curves.append(GASCurve(PI0[var,well], pressure0[var,well], rmob[var,well], sat0[var,well], ds[var,well]))
                    curves[var,well] = cur_curves
        else:
            for var in range(curves.shape[0]):
                for well in range(curves.shape[1]):
                    cur_curves = []
                    cur_curves.append(IPRCurve(PI0[var,well], pressure0[var,well], rmob[well], sat0[var,well], ds[var,well]))
                    cur_curves.append(WATCurve(PI0[var,well], pressure0[var,well], rmob[well], sat0[var,well], ds[var,well]))
                    cur_curves.append(OILCurve(PI0[var,well], pressure0[var,well], rmob[well], sat0[var,well], ds[var,well]))
                    cur_curves.append(GASCurve(PI0[var,well], pressure0[var,well], rmob[well], sat0[var,well], ds[var,well]))
                    curves[var,well] = cur_curves

        bhp_bnd = self.controller.control(curves)
        bhp_bnd = tc.from_numpy(bhp_bnd).type(tc.float32).to(device=device)
        return bhp_bnd