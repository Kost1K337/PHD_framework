"""Constants need in calculation"""

## Mathematical constants
PI = 3.14159265358979323846
G = 9.80665

## Physical constants
ATM2PA = 101325
ATM2BAR = 1.01325
BAR2PA = 100000
DAY2SEC = 24 * 3600
MM2M = 0.001
TON2KG = 1000

EPS = 1e-10
## Standard condition properties

RHO_GAS = 0.783
RHO_WAT = 1020.0
RHO_OIL = 850.0

RHO_WAT_REL = 1000
RHO_AIR_REL = 1.22381

MU_WAT = 1.1375 * 10 ** (-3) # Pa * s

## Common columns
LIQUID_RATE = "LIQUID_RATE"
WATER_CUT = "WATER_CUT"
GAS_OIL_RATIO = "GAS_OIL_RATIO"
WATER_RATE = "WATER_RATE"
PRESSURE = "PRESSURE"
CHOKE = "CHOKE"
GAS_RATE = "GAS_RATE"
OIL_RATE = "OIL_RATE"
NWATREM = "NWATREM"
NGASREM = "NGASREM"

FUNC = "func"
INV_FUNC = "inv_func"
DATA = "data"
IPR = "ipr"
XCOL = "xcol"
YCOL = "ycol"
NET = "_net"

## Network rates
NET_COLS = [
    LIQUID_RATE,
    WATER_CUT,
    GAS_OIL_RATIO,
    WATER_RATE,
    GAS_RATE,
    PRESSURE,
    CHOKE,
    NWATREM,
    NGASREM,
]


def default_curve_function(q: float, **kwargs):
    """Linear IPR function: q |-> p"""
    return kwargs["Pr"] - q / kwargs["J"]


def default_curve_inv_function(p: float, **kwargs):
    """Linear IPR function: p |-> q"""
    return kwargs["J"] * (kwargs["Pr"] - p)


def default_gor_function(p: float, **kwargs):
    """Constant GOR function"""
    _ = p
    return kwargs["GOR"]


DEFAULT_IPR_DICT = {
    FUNC: default_curve_function,
    INV_FUNC: default_curve_inv_function,
    "Pr": 181,
    "J": 3,
    "interp_dict": {
        "kind": "linear",
        "bounds_error": False,
        "fill_value": "extrapolate",
    },
}

DEFAULT_GAS_DICT = {
    FUNC: default_curve_function,
    INV_FUNC: default_curve_inv_function,
    "Pr": 181,
    "J": 3,
    "interp_dict": {
        "kind": "linear",
        "bounds_error": False,
        "fill_value": "extrapolate",
    },
}

DEFAULT_GOR_DICT = {
    FUNC: default_gor_function,
    INV_FUNC: None,
}
