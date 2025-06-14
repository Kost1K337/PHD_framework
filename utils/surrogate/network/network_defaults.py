"""Constants used by network."""
TOL_RES = 0.000003
TOL_MAS = 0.5  # sufficient for mass fraction calculation
TOL_DEB = 0.01
EKITER = 0.5
EKDAMP = 0.5
MAXITER: int = 1000
MAXITERCHOKE: int = 100
MAXITERMAIN: int = 10
TOL_CHOKE = 0.001

BHP_OFFSET = 10.0
NITER_MAS = 100
MIN_MAS = 0.00001
MIN_OIL_MASS_RATIO = 0.000001
RES_CONNECT = 0.01
MIN_TOTAL_RATE = 0.0001
EPS = 10e-16

## Network properties
MAX_NLINK = 64
START_Q = 50
MAX_Q = 1000000
MAX_DELP = 200
MAX_P = 1e10
DEFAULT_QLIQ_BOUNDS = [0, MAX_Q]
DEFAULT_CHOKE_BOUNDS = [0, MAX_DELP]


def default_ipr_function(obj, q, **kwargs):
    """Linear IPR function."""
    _ = obj
    return kwargs["Pr"] - q / kwargs["J"]


def default_ipr_inv_function(obj, P, **kwargs):
    """Linear inverse IPR function."""
    _ = obj
    return kwargs["J"] * (kwargs["Pr"] - P)


DEFAULT_IPR_DICT = {
    "func": default_ipr_function,
    "inv_func": default_ipr_inv_function,
    "Pr": 181,
    "J": 3,
    "interp_dict": {
        "kind": "linear",
        "bounds_error": False,
        "fill_value": "extrapolate",
    },
}

DEFAULT_CURVE_DICT = {"ipr": DEFAULT_IPR_DICT}
