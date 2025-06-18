"""Classes and functions related to inflow curves derived from a rate model.s"""
from typing import Any
import astropy.units as u

from .curve import Curve, get_interpolators, get_model_wrapper, CURVE_UNITS
from .. import custom_units as cu
if __name__ == 'core_engine.mipt_solver.nodal.inflow_curves.rate_model_curve':
    from core_engine.ima.curve_utils import get_last_well_value, get_bhp_limit, get_bhp_grid, sample_curve
elif __name__ == 'mipt_solver.nodal.inflow_curves.rate_model_curve':
    from ima.curve_utils import get_last_well_value, get_bhp_limit, get_bhp_grid, sample_curve

class RateModelCurve(Curve):
    """Curve based on rate model."""
    def __init__(self, params: dict, kind: str="ipr", **kwargs):
        super().__init__(params=params, kind=kind)
        self.kind = kind
        self.params = params
        self.well_name = kwargs.get("well_name", None)
        self.model = kwargs["model"]
        self.sampling_kwargs = kwargs.get("sampling_kwargs", {})
        self.interp_kwargs = kwargs.get("interp_kwargs", {})
        self.model_function = get_model_wrapper(self.kind)
    
    def update(self, well_history, **kwargs):
        """Update curve based on rate_model"""
        # get last bhp
        last_bhp = get_last_well_value(history=well_history, value="bhp")
        self.sampling_kwargs["last_bhp"] = last_bhp
        # get bhp limits
        bhp_lim = get_bhp_limit(**self.sampling_kwargs)
        self.sampling_kwargs["bhp_lim"] = bhp_lim
        # build grids based on bhp limits
        grid = get_bhp_grid(**self.sampling_kwargs)
        # create table
        curve_table = sample_curve(history=well_history, 
                             rate_model=self.model, 
                             grid=grid, 
                             kind=self.kind, 
                             well_name=self.well_name,
                             **kwargs
                             )
        
        # build interpolators
        self.direct, self.inverse = get_interpolators(curve_table,
                                                      self.model_function,
                                                      **self.interp_kwargs
                                                      )
    @staticmethod
    def from_rate_model(rate_model: Any, name: str, kind:str="ipr", **kwargs) -> 'RateModelCurve':
        if kind == "ipr":
            curve = IPRRateModelCurve(
                params={},
                model=rate_model,
                well_name=name,
                **kwargs
            )
            return curve
        elif kind in CURVE_UNITS.keys():
            curve = RateModelCurve(
                params={},
                kind=kind,
                model=rate_model,
                well_name=name,
                **kwargs
            )
            return curve
        raise KeyError(f"Curve kind `{kind}` is not supported.")

class IPRRateModelCurve(RateModelCurve):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, kind='ipr')

    def direct(self, Q):
        return super().inverse(Q)
        
    def inverse(self, Pbhp):
        return super().direct(Pbhp)