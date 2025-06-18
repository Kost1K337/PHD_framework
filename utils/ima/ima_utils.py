"""Utility function for the work of integrated model"""

import datetime as dt
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import anytree
import pandas as pd
import numpy as np

from src.mipt_solver.nodal import custom_units as cu

if __name__ == 'core_engine.ima.ima_utils':
    from mipt_solver.nodal import custom_units as cu
    from mipt_solver.nodal import Node, NodeType
elif __name__ == 'ima.ima_utils':
    from mipt_solver.nodal import custom_units as cu
    from mipt_solver.nodal import Node, NodeType

from src.mipt_solver.nodal.node import NodeType
import astropy.units as u
from .const import *  # pylint: disable=wildcard-import, unused-wildcard-import
from ._typing import OptionalPath, StrOrTimeStamp




def load_wparam(path: OptionalPath = "default_well_parameters.csv") -> pd.DataFrame:
    """Load default well parameters (mainly PI and Pres)"""
    wparam = pd.read_csv(path, index_col=0)
    wparam[WCT_COL] /= 100
    return wparam


def get_today(fstring: str = "%d.%m.%y") -> str:
    """Get current date to save the experiment data."""
    return dt.date.today().strftime(fstring)


def fprint(str_: str, **kwargs) -> None:
    """Print flush for notebook presentation."""
    print("\r", end=str_, **kwargs)


def get_last_values(
    wells: Iterable, history: pd.DataFrame, date: StrOrTimeStamp, value: str, **kwargs
) -> Dict[str, float]:
    """Get last non-zero values from dataframe before the `date` from column `value` for `well`.

    Parameters
    ----------
    wells : Iterable
        list of well names
    history : pd.DataFrame
        dataframe with **sorted** well production historical data
    date : StrORTimeStamp
        date (presumably, current date)
    value : str
        column in `history` to take values from.

    Returns
    -------
    dict
        _description_
    """
    if value == WCT_COL:
        history.loc[:, value] = history.loc[:, QWAT_COL] / (
            history.loc[:, QWAT_COL] + history.loc[:, QOIL_COL] + EPS
        )
    if value == GOR_COL:
        history.loc[:, value] = history.loc[:, QGAS_COL] / (
            history.loc[:, QOIL_COL] + EPS
        )
    if value == QLIQ_COL:
        history.loc[:, value] = history.loc[:, QWAT_COL] + history.loc[:, QOIL_COL]

    last_values = {}
    for well in wells:
        # we're interested only in one well at a time
        wdf = history[history[WELL_COL] == well]
        # we take last valid value
        last_valid_value = wdf.loc[
            (wdf[DATE_COL] <= date) & (wdf[value].notna()) & (wdf[value] > EPS), value
        ]
        last_valid_value = (
            kwargs[f"last_{value}"][well]
            if last_valid_value.size == 0
            else last_valid_value.iloc[-1]
        )
        last_values[well] = last_valid_value

    if value in (WCT_COL, GOR_COL, QLIQ_COL):
        del history[value]

    return last_values


def get_bhp_limits(
    wells: Iterable[str],
    sampling_range: Union[str, Tuple[float, ...]] = "full",
    **kwargs,
) -> Dict[str, Tuple[float, float]]:
    """Get BHP limits"""
    bhp_lim = {well: None for well in wells}

    ## take full range from to Pres
    if sampling_range == "full":
        last_bhps = kwargs["last_bhps"]
        pres = kwargs[PRES_COL]
        return {well: (0, max(pres[well], last_bhps[well])) for well in wells}

    ## make a window around last value defined in percents
    if sampling_range == "relative":
        last_bhps = kwargs["last_bhps"]
        window = kwargs.get("window", DEF_SAMPLING_WINDOW)
        bhp_lim = {}
        for well in wells:
            ll = last_bhps[well]
            bhp_lim[well] = ll * (1 + window[0] / 100), ll * (1 + window[1] / 100)
        return bhp_lim

    if isinstance(sampling_range, tuple):
        tuplen = len(sampling_range)
        if tuplen not in (1, 2):
            raise ValueError("`sampling_range` tuple should be a 1-tuple or 2-tuple")
        ## only upper bound is given, lower bound taken as 0
        if tuplen == 1:
            if sampling_range[0] < 0:
                raise ValueError("Only positive higher bound on bhp is permitted")

            return {well: (0, sampling_range[0]) for well in wells}

        if tuplen == 2:
            if sampling_range[0] >= sampling_range[1]:
                raise ValueError(
                    "`sampling_range should be sorted from lowest to highest`"
                )
            if sampling_range[0] < 0:
                raise ValueError("Only non-negative lower bound on bhp is permitted")
            return {well: sampling_range for well in wells}

    raise ValueError(
        "`sampling_range` should be a 1-tuple or 2-tuple or one of strings: `full`, `relative`"
    )


def get_grids(
    wells: Iterable[str], sampling_mode: str = "equal", **kwargs
) -> Dict[str, np.ndarray]:
    """Construct grids of BHP."""
    ## even grid: limits are given and points are sampled equidistantly to provide constant density `bhp_dens`
    if sampling_mode == "equal":
        grids = {}
        bhp_dens = kwargs.get("bhp_dens", DEF_BHP_GRID_DENSITY)
        bhp_lim = kwargs["bhp_lim"]
        for w in wells:
            lim = bhp_lim[w]
            # print(lo_limit, hi_limit)
            n_bhp = round(bhp_dens * (lim[1] - lim[0]))
            grids[w] = np.linspace(lim[0], lim[1], num=n_bhp)
        return grids
    ## uneven grid: points are concentrated around a given point and sparce far from it
    if sampling_mode == "concentrated":
        raise NotImplementedError("`sampling` mode `concentrated` is not implemented")

    raise ValueError(
        "`sampling_mode` should be one of the following: `equal`, `concentrated`."
    )


def get_curves(
    wells: Iterable[str],
    history: pd.DataFrame,
    sim: Any,
    grids: Dict[str, np.ndarray],
    **kwargs,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Sample IPR (GOR, WCT...) curves using grids and rate model (sim)"""
    dictnames = (IPR, WCT_COL, GOR_COL, QGAS_COL, QOIL_COL, QWAT_COL)
    dicts = {n: {} for n in dictnames}

    for well in wells:
        grid = grids[well]

        # fill second columns of `DATA` array with BHP grid
        for dct in dicts.values():
            dct[well] = {}
            dct[well][DATA] = np.empty((len(grid), 2))
            dct[well][DATA][:, 1] = grid
        # sample rate model with bhp points and save out to first column of resp. array
        for i, bhp in enumerate(grid):
            w_pred = sim({well: bhp}, history, *kwargs[SIM_ARGS])[well]
            qoil, qwat, qgas = w_pred[QOIL_COL], w_pred[QWAT_COL], w_pred[QGAS_COL]
            qliq = qoil + qwat
            dicts[IPR][well][DATA][i, 0] = qliq
            dicts[WCT_COL][well][DATA][i, 0] = qwat / (qliq + EPS)
            dicts[QGAS_COL][well][DATA][i, 0] = qgas
            dicts[GOR_COL][well][DATA][i, 0] = qgas / (qoil + EPS)
            dicts[QOIL_COL][well][DATA][i, 0] = qoil
            dicts[QWAT_COL][well][DATA][i, 0] = qwat

    return dicts


def process_curves(dct: dict, val: str = IPR) -> Tuple[dict, List[str]]:
    """Process IPR (GOR, WCT) curves and return wells for which curves are invalid

    Parameters
    ----------
    dct : dict
        dictionary containing curve data for every well
    val : str, optional
        curve type, by default IPR

    Returns
    -------
    Tuple[dict,List[str]]
        tuple of processed curve dictionary and list of "invalid" wells
    """
    dct_out = deepcopy(dct)
    swells = []
    for wname, wdict in dct.items():
        dd = wdict[DATA]
        ys = dd[:, 0]
        ps = dd[:, 1]

        ## checked always, expected rarely
        if np.allclose(ps - ps[0], 0, atol=EPS):
            swells.append(wname)
            continue

        if val in (IPR, GOR_COL):
            if np.allclose(ys - ys[0], 0, atol=EPS):
                swells.append(wname)
                continue

        if val == IPR:
            # if slope is negative filter
            ys_dim1 = np.array((ys[1:] - ys[:-1] < 0).tolist() + [False])
            ys_dim2 = np.array((ys[2:] - ys[:-2] < 0).tolist() + [False, False])
            ys = ys[ys_dim1 & ys_dim2]
            ps = ps[ys_dim1 & ys_dim2]

        # ys should be positive
        ys_dim = ys > 0
        ys = ys[ys_dim]
        ps = ps[ys_dim]

        if val == WCT_COL:
            ys = np.clip(ys, 0, 1)

        dd = np.hstack((ys.reshape(-1, 1), ps.reshape(-1, 1)))
        if dd.shape[0] == 1:
            # print("Invalid IPR curve well {}: 1 pair is valid: y={},p={}".format(wname, ys[0], ps[0]))
            swells.append(wname)
            continue
        if dd.size == 0:
            # print("Invalid IPR curve well {}: in all pairs either y or p is zero.".format(wname))
            swells.append(wname)
            continue
        dct_out[wname][DATA] = dd

    for k in swells:
        del dct_out[k]

    return dct_out, swells


def linear_ipr(p, **kwargs):
    """Inverse IPR function"""
    return kwargs["J"] * (kwargs["Pr"] - p)


def default_ipr(object, q, **kwargs):
    """Linear IPR function."""
    _ = object
    return kwargs["Pr"] - q / kwargs["J"]


def replace_invalid_ipr_curves(dct, swells, wparam, **kwargs):
    out_dict = deepcopy(dct)
    for well in swells:
        dct_ = {
            "Pr": wparam.loc[well, PRES_COL],
            "func": kwargs.get("default_ipr", default_ipr),
            "J": wparam.loc[well, "pi"],
        }
        if not kwargs.get("tosample", False):
            out_dict[well] = dct_
            continue

        grid = kwargs["grids"][well]
        sample_func = kwargs.get("sample_func", linear_ipr)
        qliq = sample_func(grid, **{k: v for k, v in dct.items() if k != FUNC})
        out_dict[well] = {}
        out_dict[well][DATA] = np.empty((len(grid), 2))
        out_dict[well][DATA][:, 1] = grid
        out_dict[well][DATA][:, 0] = qliq
    return out_dict


def replace_wct_with_const(wct_dct: dict, last_wcts: dict) -> dict:
    """Replace wct curve with last known historical values."""
    wct_dict = deepcopy(wct_dct)
    for well, wdata in wct_dict.items():
        wdata[DATA][:, 0] = last_wcts[well]
    return wct_dict


def replace_invalid_gor_wct_curves(dct: dict, swells: list, val: str, **kwargs):
    """Replace invalid GOR, WCT curves with constant values taken from history."""
    out_dict = deepcopy(dct)

    def const_last_val(x, **kwargs):
        """Return last value in shape of input"""
        x_0 = np.asarray_chkfinite(x)
        return np.ones_like(x_0) * kwargs[f"last_{val}"]

    last_vals = kwargs.get("last_vals", None)
    if last_vals is None:
        raise ValueError(
            "Invalid 'wct','gor' curves are replaced by const function but `last_vals` not given"
        )

    for well in swells:
        out_dict[well] = {
            f"last_{val}": last_vals[well],
            "func": const_last_val,
        }
    return out_dict


def replace_invalid_curves(
    dct: dict, swells: List[str], wparam: pd.DataFrame, val: str = QLIQ_COL, **kwargs
) -> Dict[str, dict]:
    """Replace curves for wells in `swells` by params from `wparam` of `last_vals`

    Parameters
    ----------
    dct : dict
        dictionary containing curve data for every well
    swells : List[str]
        list of wells to replace curve dict
    wparam : dict
        dataframe with well names in index and columns `pres` and `pi`
    val : str, optional
        curve type, by default `qliq` (`wct`, `gor`, `qliq`, `ipr`)

    Returns
    -------
    Dict[str,dict]
        dictionary containing curve data for every well

    Raises
    ------
    ValueError
        Raised when `val` is not supported as a curve determination
    ValueError
        Raised when curve is GOR, WCT but `last_vals` keyword is not given
    """

    if val in (IPR, QLIQ_COL):
        return replace_invalid_ipr_curves(dct, swells, wparam, **kwargs)

    if val in (WCT_COL, GOR_COL):
        return replace_invalid_gor_wct_curves(dct, swells, val, **kwargs)

    raise ValueError(
        f"`val` supports only values: `{WCT_COL}`, `{GOR_COL}`, `{IPR}`, `{QLIQ_COL}`. Value `{val}` given"
    )


def get_rate_dict_from_history(
    wells: List[str], hist: pd.DataFrame, date: StrOrTimeStamp
) -> Dict[str, Dict[str, float]]:
    """Form dictionary with rate parameters from history

    Parameters
    ----------
    wells : List[str]
        list with well names
    hist : pd.DataFrame
        dataframe with historical data
    date : StrORTimeStamp
        date (presumably, current date in forecast)

    Returns
    -------
    Dict[str,Dict[str,float]]
        keywords for rate parameters comply with `deepfield.Network`
    """
    dct = {}
    for well in wells:
        ind = hist.loc[
            (
                hist[WELL_COL].isin([well])
                & (hist[DATE_COL] <= date)
                & hist[QOIL_COL].notna()
                & hist[QWAT_COL].notna()
                & hist[QGAS_COL].notna()
            )
        ].index[-1]
        qoil = hist.loc[ind, QOIL_COL]
        qwat = hist.loc[ind, QWAT_COL]
        qgas = hist.loc[ind, QGAS_COL]
        qliq = qoil + qwat

        dct[well] = {}
        dct[well][LIQUID_RATE] = qliq
        dct[well][GAS_OIL_RATIO] = qgas / (qoil + EPS)
        dct[well][WATER_CUT] = qwat / (qliq + EPS)

    return dct


def get_rate_dict_from_last(
    wells: List[str],
    last_liqs: Dict[str, float],
    last_wcts: Dict[str, float],
    last_gors: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Form dictionary from `last_x` dictionaries."""
    rate_dict = {}
    for well in wells:
        rate_dict[well] = {}
        rate_dict[well][LIQUID_RATE] = last_liqs[well]
        rate_dict[well][WATER_CUT] = last_wcts[well]
        rate_dict[well][GAS_OIL_RATIO] = last_gors[well]
    return rate_dict


def initialize_network(
    ws: Any, ipr_dict: dict, wct_dict: dict, gor_dict: dict, **kwargs
) -> Any:
    """Fill `deepfield.Network` node values with curves, optionally with rates and sink pressure

    Parameters
    ----------
    ws : deepfield.Network
        deepfield.Network object
    ipr_dict : dict
        dictionary with IPR curve data for every well
    wct_dict : dict
        dictionary with WCT curve data for every well
    gor_dict : dict
        dictionary with GOR curve data for every well

    Returns
    -------
    deepfield.Network
        deepfield.Network object
    """
    keys = (IPR, WCT_COL, GOR_COL)
    dicts = (ipr_dict, wct_dict, gor_dict)
    for key, dct in zip(keys, dicts):
        interp_kwargs = kwargs.get(
            f"interp_dict_{key}",
            {
                "kind": "linear",
                "bounds_error": False,
                "fill_value": "extrapolate",
            },
        )

        for well in dct:
            if "interp_dict" in dct[well]:
                dct[well]["interp_dict"].update(interp_kwargs)
            else:
                dct[well]["interp_dict"] = interp_kwargs

    ws._init_network()
    ws._set_ipr(ipr_dict)
    ws._set_wct(wct_dict)
    ws._set_gor(gor_dict)
    if "rate_dict" in kwargs:
        ws._set_rates(kwargs["rate_dict"])

    if "sk_press" in kwargs:
        ws._sink.press = kwargs["sk_press"]

    return ws


def initialize_network_mipt(
    ws: Any, ipr_dict: dict, wct_dict: dict, gor_dict: dict, **kwargs
) -> Any:
    """Fill `Network` node values with curves, optionally with rates and sink pressure

    Parameters
    ----------
    ws : nodal.Network
        nodal.Network object
    ipr_dict : dict
        dictionary with IPR curve data for every well
    wct_dict : dict
        dictionary with WCT curve data for every well
    gor_dict : dict
        dictionary with GOR curve data for every well

    Returns
    -------
    nodal.Network
        nodal.Network object
    """
    keys = (IPR, WCT_COL, GOR_COL)
    dicts = (ipr_dict, wct_dict, gor_dict)
    for key, dct in zip(keys, dicts):
        interp_kwargs = kwargs.get(
            f"interp_dict_{key}",
            {
                "kind": "linear",
                "bounds_error": False,
                "fill_value": "extrapolate",
            },
        )

        for well in dct:
            if "interp_dict" in dct[well]:
                dct[well]["interp_dict"].update(interp_kwargs)
            else:
                dct[well]["interp_dict"] = interp_kwargs
    ws.set_curves(ipr_dict, "ipr")
    ws.set_curves(wct_dict, "wct")
    ws.set_curves(gor_dict, "gor")
    if "rate_dict" in kwargs:
        ws._set_rates(kwargs["rate_dict"])

    if "pres_dict" in kwargs:
        ws.update_well_respres(kwargs["pres_dict"])

    if "sk_press" in kwargs:
        ws.sink.p = kwargs["sk_press"]

    return ws


def generate_network_bounds(
    ws: Any, mode: str = IPR, **kwargs
) -> List[Tuple[float, float]]:
    """Generate a list of tuples of two floats

    Parameters
    ----------
    ws : Any
        deepfield.Network object
    mode : str, optional
        mode of generating bounds, by default 'ipr' ('prev','default')

    Returns
    -------
    List[Tuple[float,float]]
        list of bounds for every position
    """
    if mode == "default":
        return [DEF_RATE_BOUNDS] * len(ws.wellnames) + [DEF_CHOKE_BOUNDS] * len(
            ws._autochokes
        )

    if mode == "prev":
        window = kwargs.get("window", DEF_PERCENT_BOUNDS)
        bounds = []
        for w in ws.wellnames:
            ll = kwargs["last_qliqs"][w]
            lb = (1 + window[0] / 100) * ll
            lb = max(lb, 0)
            hb = (1 + window[1] / 100) * ll
            bounds.append((lb, hb))

        bounds.extend(
            [DEF_CHOKE_BOUNDS] * len(ws._autochokes)
        )  # pylint: disable=protected-access
        return bounds

    if mode == IPR:
        ipr_dict = kwargs["ipr_dict"]
        bounds = []

        for w, wdata in ipr_dict.items():
            qmax = (
                wdata[DATA][0, 0] if DATA in wdata else (wdata["Pr"] - 1) * wdata["J"]
            )
            qmin = 0
            bounds.append((qmin, qmax))

        bounds.extend(
            [DEF_CHOKE_BOUNDS] * len(ws._autochokes)
        )  # pylint: disable=protected-access
        return bounds


def generate_network_bounds_mipt(
    ws: Any, mode: str = IPR, **kwargs
) -> List[Tuple[float, float]]:
    """Generate a list of tuples of two floats

    Parameters
    ----------
    ws : Any
        nodal.Network object
    mode : str, optional
        mode of generating bounds, by default 'ipr' ('prev','default')

    Returns
    -------
    List[Tuple[float,float]]
        list of bounds for every position
    """
    if mode == "default":
        return [DEF_RATE_BOUNDS] * len(ws.wellnames) + [DEF_CHOKE_BOUNDS] * len(
            ws.autochokes
        )

    if mode == "prev":
        window = kwargs.get("window", DEF_PERCENT_BOUNDS)
        bounds = []
        for w in ws._wellnames:
            ll = kwargs["last_qliqs"][w]
            lb = (1 + window[0] / 100) * ll
            lb = max(lb, 0)
            hb = (1 + window[1] / 100) * ll
            bounds.append((lb, hb))

        bounds.extend([DEF_CHOKE_BOUNDS] * len(ws.autochokes))
        return bounds

    if mode == IPR:
        ipr_dict = kwargs["ipr_dict"]
        bounds = []

        for w, wdata in ipr_dict.items():
            qmax = (
                wdata[DATA][0, 0] if DATA in wdata else (wdata["Pr"] - 1) * wdata["J"]
            )
            qmin = 0
            bounds.append((qmin, qmax))

        bounds.extend([DEF_CHOKE_BOUNDS] * len(ws.autochokes))
        return bounds


def solve_network(wells: Any, **net_kwargs) -> Tuple[Any, np.array, float]:
    """Solve deepfield.Network using multiple attempts"""
    print("Solving network...")
    n_trials = net_kwargs.pop("n_trials", 20)
    n_errors = net_kwargs.pop("n_errors", 10)
    n_retrials = net_kwargs.pop("n_retrials", 1)
    max_error = net_kwargs.pop("max_error", 100)
    min_error = net_kwargs.pop("min_error", 1)
    guess_0 = net_kwargs.pop("guess_0", None)
    bounds = net_kwargs.pop("bounds", None)
    method = net_kwargs.pop(METHOD, "SLSQP")
    # opt_value = net_kwargs.pop("opt_value", QLIQ_COL)

    i = 1
    j = 1
    ei = 0
    le = None
    best_err = np.inf
    best_sol = None

    while i <= n_trials:
        fprint(f"NETWORK ATTEMPT {i}\n")

        if ei > n_errors:
            raise le

        # print(f"Attempt {i}")
        try:
            wells.total_network_analysis(
                guess_0=guess_0,
                bounds=bounds,
                method=method,
                # opt_value=opt_value,
                **net_kwargs,
            )
        except Exception as e:
            le = e
            print(e)
            e_i += 1
            continue

        err = wells.calculate_network_error(
            wells._network_solution
        )  # pylint: disable=protected-access
        if np.any(np.isnan(err)):
            print("The error function is nan!")
            error_iter += 1
            continue
        if err < best_err:
            best_err = err
            best_sol = wells._network_solution  # pylint: disable=protected-access

        if err < min_error:
            best_err = err
            best_sol = wells._network_solution  # pylint: disable=protected-access
            break

        if i == n_trials and best_err > max_error:
            print(
                f"Made last attempt but tolerance is not reached: {best_err} > {max_error}. Starting all over again."
            )
            i = 0

            if j == n_retrials:
                print(f"Exhausted number of retrials: {n_retrials}")
                break

            j += 1
        i += 1

    if best_sol is None or best_err > max_error:
        warnings.warn("Optimizer didn't converge!", UserWarning)

    return wells, best_sol, best_err

class NetworkSolutionError(Exception):
    "Raised when something goes wrong with solution of network."

def solve_network_mipt(solver: Any, **net_kwargs):
    """Solve Network using `nodal` solver."""
    print("Solving network...")
    n_trials = net_kwargs.pop("n_trials", 20)
    n_errors = net_kwargs.pop("n_errors", 10)
    n_retrials = net_kwargs.pop("n_retrials", 1)
    max_error = net_kwargs.pop("max_error", 100)
    min_error = net_kwargs.pop("min_error", 1)
    raise_errors = net_kwargs.pop("raise_errors", False)
    # opt_value = net_kwargs.pop("opt_value", QLIQ_COL)

    i = 1
    j = 1
    error_iter = 0
    last_error = None
    best_err = np.inf
    best_sol = None

    while i <= n_trials:
        fprint(f"NETWORK ATTEMPT {i}\n")

        if error_iter > n_errors:
            raise last_error

        # print(f"Attempt {i}")
        try:
            solver.solve(**net_kwargs)
        except Exception as e:
            if raise_errors:
                raise NetworkSolutionError("Something went wrong!") from e
            last_error = e
            print(e)
            error_iter += 1
            continue

        err = solver.nodal_loss
        if np.any(np.isnan(err)):
            print("The error function is nan!")
            error_iter += 1
            continue
        if err < best_err:
            best_err = err
            best_sol = solver.network_solution

        if err < min_error:
            best_err = err
            best_sol = solver.network_solution
            break

        if i == n_trials and best_err > max_error:
            print(
                "\nMade last attempt but tolerance is not reached: {} > {}. Starting all over again.".format(
                    best_err, max_error
                )
            )
            i = 0

            if j == n_retrials:
                print(f"\nExhausted number of retrials: {n_retrials}")
                break

            j += 1
        i += 1

    if best_sol is None or best_err > max_error:
        print(RuntimeWarning("Optimizer didn't converge!"))

    return solver, best_sol, best_err


def predict_rates(
    wells: List[str],
    date: StrOrTimeStamp,
    net_rates: Dict[str, Dict[str, float]],
    # history: pd.DataFrame,
    mode: str = "rate_model",
    **kwargs,
) -> pd.DataFrame:
    if mode not in ("network", "net", "rate_model", "model"):
        raise ValueError(f"Prediction mode `{mode}` not supported.")

    # net_bhps = {w.name: net_rates[w.name][PRESSURE] for w in wells}
    # pred = sim(net_bhps, history, kwargs[INJT_COL])

    df = {}
    for i, w in enumerate(wells):
        if w in net_rates:
            df[i] = {}
            df[i][DATE_COL] = date
            df[i][WELL_COL] = w
            # df[i][INJT_COL] = kwargs[INJT_COL]
            # df[i][PRES_COL] = w.b_liq

            # if mode in ("network", "net"):
            df[i][QLIQ_COL] = net_rates[w][LIQUID_RATE]
            df[i][QOIL_COL] = net_rates[w][LIQUID_RATE] - net_rates[w][WATER_RATE]
            df[i][QWAT_COL] = net_rates[w][WATER_RATE]
            df[i][QGAS_COL] = net_rates[w][GAS_RATE]
            df[i][WBHP_COL] = net_rates[w][PRESSURE]

    df = pd.DataFrame(df).T
    ## negative values are nullified for these columns
    for col in (QOIL_COL, QWAT_COL, QGAS_COL, WBHP_COL):
        df[col] = np.maximum(df[col], 0)
    df[WELL_COL] = df[WELL_COL].astype(str)
    return df


def generate_first_guess(
    wells: Iterable, hist: pd.DataFrame, opt_value: str = QLIQ_COL
) -> List[float]:
    guess = hist.loc[
        hist[DATE_COL] == hist[DATE_COL].max(), [WELL_COL, QOIL_COL, QWAT_COL, WBHP_COL]
    ].set_index(WELL_COL)
    guess[QLIQ_COL] = guess[QWAT_COL] + guess[QOIL_COL]
    return [guess.loc[w, opt_value] for w in wells]


def detach_wells(wells: Any, wells_to_detach: list) -> Tuple[Any, dict]:
    """Detach branches attached only to wells which are in `wells_to_detached`.

    Parameters
    ----------
    wells : Union[Wells,deepfield.Network]
        deepfield.Network object
    wells_to_detach : list
        wells to detach

    Returns
    -------
    tuple
        tuple of deepfield.Network object and list of detached nodes.
    """
    detached = {}
    ## assign to all wells to detach attribute detach with value 1
    for well in wells:
        setattr(well, DETACH, 1 if well.name in wells_to_detach else 0)
    ## propagate the attribute up the network to determine branches to detach
    wells._determine_detached()
    ## start from the root and detach branches until there are no nodes with detach=1 and save to dict
    while sum(wells._select_nodes(lambda n: DETACH in n.attributes, attr="detach")):
        for node in anytree.LevelOrderIter(wells.root):
            if DETACH not in node.attributes:
                continue
            if node.detach:
                print(f"DETACHING BRANCH {node.name}")
                detached[node.name] = {"parent": node.parent.name, "branch": node}
                node.parent = None
                break
    ## delete detach attribute: no longer needed
    for node in anytree.LevelOrderIter(wells.root):
        if DETACH not in node.attributes:
            continue
        delattr(node, DETACH)

    return wells, detached


def reattach_wells(wells: Any, detached: dict) -> Any:
    """Reattach branches with non-functioning wells to tree.

    Parameters
    ----------
    wells : deepfield.Network
        deepfield.Network object
    detached : dict
        dictionary with info to reconnect nodes

    Returns
    -------
    deepfield.Network
        deepfield.Network object with reattached branches
    """
    for data in detached.values():
        parent = data["parent"]
        branch = data["branch"]
        print("Reattaching branch {} to parent {}".format(branch.name, parent))
        branch.parent = wells[parent]
    return wells


def init_network_for_validation(wells: Any, rate_dict: dict, **kwargs) -> Any:
    sink = kwargs.get("sink", None)
    # sink = "MBSU"
    wells._init_network()
    wells._set_rates(rate_dict)
    wells._set_sink(sink)

    if "sk_press" in kwargs:
        wells._sink.press = kwargs["sk_press"]

    return wells


def update_curves(dicts: tuple, out_dict: dict, date: str) -> None:
    keys = (IPR, WCT_COL, GOR_COL, QOIL_COL, QWAT_COL, QGAS_COL)

    for key, dct in zip(keys, dicts):
        out_dict[key][date] = dct


def fill_wellnodes(net: Any, **kwargs) -> None:
    """Fill wellnodes with data."""
    if "last_wcts" in kwargs:
        wcts = kwargs["last_wcts"]
        for well in net.wellnodes:
            well.wct = wcts[well.name] * cu.CUB_CUB

    if "last_gors" in kwargs:
        gors = kwargs["last_gors"]
        for well in net.wellnodes:
            well.gor = gors[well.name] * cu.CUB_CUB


def get_delta_time(
    t0: float, t1: float, n_steps: int
) -> Tuple[dt.timedelta, dt.timedelta]:
    total_time = dt.timedelta(seconds=t1 - t0)
    time_per_iter = total_time / n_steps

    return total_time, time_per_iter

class NotDCACurveError(Exception):
    """Raised when code tries to update not-DCA curve as DCA-curve"""

def update_dca_curves_parameters(network: Any, 
                                 cur_date: StrOrTimeStamp, 
                                 last_date: StrOrTimeStamp,
                                 factor: Optional[Dict[str, float]]=None) -> None:
    if factor is None:
        factor = {}
        
    for node in network.nodes:
        for curve in (node.ipr_curve, node.wat_curve, node.oil_curve, node.gas_curve):
            if curve is None:
                continue
            try:
                working_time = (pd.to_datetime(cur_date) - pd.to_datetime(last_date)).days * u.day
                working_time_with_factor = working_time * factor.get(node.name, WORK_FACTOR)
                curve.update_cum(working_time_with_factor)
                curve.update_ipr()
            except AttributeError as e:
                raise NotDCACurveError("Trying to update DCA curve parameters not in DCA curves") from e

def calculate_monthly_rates_from_pred(net_rates: Dict[str, Dict[str, float]], 
                                      cur_date: StrOrTimeStamp, 
                                      last_date: StrOrTimeStamp,
                                      factor: Dict[str, float]=None) -> Dict[str, Dict[str, float]]:
    """Calculate monthly rates using number of days in a period and well working factor."""
    if factor is None:
        factor = {}
        
    rate_cols = [LIQUID_RATE, OIL_RATE, WATER_RATE, GAS_RATE]
    num_days = (pd.to_datetime(cur_date) - pd.to_datetime(last_date)).days
    out = deepcopy(net_rates)
    for well in out.keys():
        for col in rate_cols:
            out[well][col] *= num_days * factor.get(well, WORK_FACTOR)
    return out

def set_dca_rate_pressure(network: Any, net_rates: Dict[str, Dict[str, float]]) -> None:
    """Update rate and pressure at DCA curves."""
    
    for node in network.nodes:
        for curve, column in zip((node.ipr_curve, node.wat_curve, node.oil_curve, node.gas_curve), 
                         (LIQUID_RATE, OIL_RATE, WATER_RATE, GAS_RATE)):
            if curve is None:
                continue

            curve.Q_prev = net_rates[node.name][column] * cu.CUB
            curve.Pbhp_prev = net_rates[node.name][PRESSURE] * u.bar

def set_rate_model_curves( # TODO: должна принимать ИМА (Network) и всем узлам с типом SOURCE присуждать кривые
    nodes: Iterable,
    rate_model: Any, # TODO: абстрактный класс, возращающий Curve, что иннополису дал посмотреть
    **kwargs
) -> None:
    for node in nodes:
        if node.ntype != NodeType.SOURCE:
            continue
        node.set_rate_model_phase_curves(rate_model, **kwargs)
