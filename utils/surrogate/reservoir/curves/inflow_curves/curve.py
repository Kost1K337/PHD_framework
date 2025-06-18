"""Curve Module."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple, Union
from functools import partial

import numpy as np
from ..const import *  # pylint: disable=wildcard-import,unused-wildcard-import

from .. import custom_units as cu
from astropy import units as u

import scipy.interpolate as intp


class NotEnoughInfoError(Exception):
    """Raise when there is not enough info infer maximal liquid rate"""


class CurveEvaluationError(Exception):
    """Raised when there is problem in curve code."""


CURVE_UNITS = {
    "ipr": {"direct": u.bar, "inverse": cu.CUB},
    "wct": {"direct": cu.CUB_CUB, "inverse": u.bar},
    "gor": {"direct": cu.CUB_CUB, "inverse": u.bar},
    "wat": {"direct": cu.CUB, "inverse": u.bar},
    "gas": {"direct": cu.CUB, "inverse": u.bar},
    "oil": {"direct": cu.CUB, "inverse": u.bar},
}


def model_func(x, model, **kwargs):
    _ = kwargs
    return model(x)


def wct_model_func(x, model, **kwargs):
    _ = kwargs
    out = model(x)
    out = np.clip(out, 0, 1)
    return out

def get_interpolators(data, model_wrapper, **interp_kwargs):
    
    model = intp.interp1d(data[:, 1], data[:, 0], **interp_kwargs)
    inv_model = intp.interp1d(data[:, 0], data[:, 1], **interp_kwargs)
    func = partial(model_wrapper, model=model)
    inv_func = partial(model_wrapper, model=inv_model)
    return func, inv_func

def get_model_wrapper(kind="ipr"):
    return wct_model_func if kind == "wct" else model_func

class BaseCurve(ABC):
    """Base class for curve."""

    params: Dict[str, Any] = None
    direct: Union[Callable, None] = None
    inverse: Union[Callable, None] = None

    @abstractmethod
    def func() -> u.Quantity:
        """Direct evaluation for curve."""

    @abstractmethod
    def inv_func() -> u.Quantity:
        """Inverse evaluation for curve."""


class Curve(BaseCurve):
    """Curve object."""

    def __init__(self, params: dict, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.from_unit = CURVE_UNITS[self.kind]["inverse"]
        self.to_unit = CURVE_UNITS[self.kind]["direct"]
        if DATA in params:
            model_function = get_model_wrapper(self.kind)
            data = params.pop(DATA)
            self.direct, self.inverse = get_interpolators(data, 
                                                          model_function, 
                                                          **params["interp_dict"]
                                                          )
        if FUNC in params:
            self.direct = params.pop(FUNC)
        if INV_FUNC in params:
            self.inverse = params.pop(INV_FUNC)

        self.params = params

    def func(self, x: u.Quantity) -> u.Quantity:
        if not hasattr(x, "unit"):
            x = x * self.from_unit
        inp = x if x.unit == self.from_unit else x.to(self.from_unit)
        try:
            out = self.direct(inp, **self.params)
        except u.UnitConversionError as e:
            raise CurveEvaluationError(
                f"There is a problem evaluating {self.kind} direct function : {self.params}"
            ) from e
        if not hasattr(out, "unit"):
            out = out * self.to_unit
        assert out.unit == self.to_unit, "Output unit is not correct!"
        return out

    @staticmethod
    def from_p_pi(p: u.Quantity, pi: u.Quantity, kind: str) -> 'Curve':
        curve = Curve({}, kind)
        curve.direct = lambda delta_p: p + pi * delta_p
        if pi != 0:
            curve.inverse = lambda q: (q - p) / pi
        else:
            curve.inverse = lambda _: 0


        return curve
    

    @staticmethod
    def from_fraction(a1: float, a2, b1: float, b2: float, kind: str) -> 'Curve':
        curve = Curve({}, kind)
        curve.direct = lambda delta_p: (a1 * delta_p + b1) / (a2 * delta_p + b2)
        curve.inverse = lambda q: (b1 - b2) / (q * a2 - 1)

        return curve


    def __call__(self, *arg, **kwargs) -> u.Quantity:
        return self.func(*arg, **kwargs)

    def inv_func(self, x: u.Quantity) -> u.Quantity:
        if not hasattr(x, "unit"):
            x = x * self.to_unit
        inp = x if x.unit == self.to_unit else x.to(self.to_unit)
        try:
            out = self.inverse(inp, **self.params)
        except u.UnitConversionError as e:
            raise CurveEvaluationError(
                f"There is a problem evaluating {self.kind} inverse function : {self.params}"
            ) from e

        if not hasattr(out, "unit"):
            out = out * self.from_unit
        assert out.unit == self.from_unit, "Output unit is not correct!"
        return out

    @property
    def qmax(self):
        """Find maximal rate."""
        if "Pr" in self.params and "J" in self.params:
            return self.params["Pr"] * self.params["J"]
        else:
            try:
                out = self.inv_func(0) if self.kind == "ipr" else self.func(0)
                return out
            except Exception as e:
                raise NotEnoughInfoError(
                    "Not enough info or problem to infer maximal rate."
                ) from e
