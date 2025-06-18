"""Utility functions connected to rate-model-based inflow curves."""

from typing import Any, Dict, Iterable, Optional, Tuple, Union
import numpy as np

import pandas as pd
from astropy import units as u

from .const import * # pylint=disable=wildcard-import
from ._typing import StrOrTimeStamp


def get_last_well_value(
    history: pd.DataFrame, 
    value: str, 
    date: Optional[StrOrTimeStamp]=None, 
    **kwargs
) -> Union[float,u.Quantity]:
    """Get last non-zero values from dataframe before the `date` from column `value` for `well`.

    Parameters
    ----------
    history : pd.DataFrame
        dataframe with **sorted** well production historical data
    value : str
        column in `history` to take values from.
    date : StrORTimeStamp
        date (presumably, current date)

    Returns
    -------
    Union[float,u.Quantity]
        last found or provided value
    """
    date_col = history[DATE_COL]
    ## the date is taken as last
    if date is None:
        date = date_col.iloc[-1]
    
    ## calculate water cut
    if value == WCT_COL:
        data_col = history.loc[:, QWAT_COL] / (
            history.loc[:, QWAT_COL] + history.loc[:, QOIL_COL] + EPS
        )
    ## calculate gor
    elif value == GOR_COL:
        data_col = history.loc[:, QGAS_COL] / (
            history.loc[:, QOIL_COL] + EPS
        )
    ## calculate liquid rate
    elif value == QLIQ_COL:
        data_col = history.loc[:, QWAT_COL] + history.loc[:, QOIL_COL]
    else:
        data_col = history[value]
    ## data
    data = pd.DataFrame({DATE_COL:date_col, value:data_col})
    ## last valid values
    last_valid_values = data.loc[
        (data[DATE_COL] <= date) & (data[value].notna()) & (data[value] > EPS), value
    ]
    ## last valid value
    last_valid_value = (
        kwargs.get("default", None)
        if last_valid_values.size == 0
        else last_valid_values.iloc[-1]
    )

    return last_valid_value

def get_bhp_limit(
    sampling_range: Tuple[float, ...]="full",
    **kwargs,
) -> Tuple[float, float]:
    """Get BHP limits"""

    ## take full range from to Pres
    if sampling_range == "full":
        pres = kwargs.get(PRES_COL, None)
        if pres is None:
            return DEF_PRES_BOUNDS
        return (0, pres)

    ## make a window around last value defined in percents
    if sampling_range == "relative":
        ll = kwargs["last_bhp"]
        window = kwargs.get("window", DEF_SAMPLING_WINDOW)
        bhp_lim = ll * (1 + window[0] / 100), ll * (1 + window[1] / 100)
        return bhp_lim

    if isinstance(sampling_range, tuple):
        tuplen = len(sampling_range)
        if tuplen not in (1, 2):
            raise ValueError("`sampling_range` tuple should be a 1-tuple or 2-tuple")
        ## only upper bound is given, lower bound taken as 0
        if tuplen == 1:
            if sampling_range[0] < 0:
                raise ValueError("Only positive higher bound on bhp is permitted")

            return 0, sampling_range[0]

        if tuplen == 2:
            if sampling_range[0] >= sampling_range[1]:
                raise ValueError(
                    "`sampling_range should be sorted from lowest to highest`"
                )
            if sampling_range[0] < 0:
                raise ValueError("Only non-negative lower bound on bhp is permitted")
            return sampling_range

    raise ValueError(
        "`sampling_range` should be a 1-tuple or 2-tuple or one of strings: `full`, `relative`"
    )
    
def get_bhp_grid(
    sampling_mode: str = "equal",
    **kwargs
) -> np.ndarray:
    """Construct BHP grid."""
    ## equidistant grid: points are sampled with density `bhp_dens` between two points `bhp_lim`
    if sampling_mode == "equal":
        bhp_dens = kwargs.get("bhp_dens", DEF_BHP_GRID_DENSITY)
        bhp_lim = kwargs["bhp_lim"]
        n_bhp = round(bhp_dens * (bhp_lim[1] - bhp_lim[0]))
        grid = np.linspace(bhp_lim[0], bhp_lim[1], num=n_bhp)
        return grid

    ## uneven grid: points are concentrated around a given point and sparce far from it
    if sampling_mode == "concentrated":
        raise NotImplementedError("`sampling` mode `concentrated` is not implemented")

    raise ValueError(
        "`sampling_mode` should be one of the following: `equal`, `concentrated`."
    )
    
def get_value_from_pred(
    pred: Dict[str, float], 
    kind: str="ipr"
) -> float:
    """Return appropriate value from rate model prediction.

    Parameters
    ----------
    pred : Dict[str, float]
        dictionary with predicted rates
    kind : str, optional
        curve kind, by default "ipr"

    Returns
    -------
    float
        float value

    Raises
    ------
    ValueError
        Provided key is not correct.
    """
    qoil, qwat, qgas = pred[QOIL_COL], pred[QWAT_COL], pred[QGAS_COL]
    qliq = qoil + qwat
    wct = qwat / (qliq + EPS)
    gor = qgas / (qoil + EPS)
    if kind in (IPR, QLIQ_COL):
        return qliq
    if kind == WCT_COL:
        return wct
    if kind == GOR_COL:
        return gor
    if kind in (QWAT_COL, "wat"):
        return qwat
    if kind in (QOIL_COL, "oil"):
        return qoil
    if kind in (QGAS_COL, "gas"):
        return qgas
    raise ValueError(f"Value `{kind}` is not correct.")
    
def sample_curve(
    history: pd.DataFrame,
    rate_model: Any,
    grid: np.ndarray,
    kind:str="ipr",
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Sample curve of `kind` using `grid` and `rate_model`"""

    # fill second columns of `DATA` array with BHP grid
    data = np.empty((len(grid), 2))
    data[:, 1] = grid
    # sample rate model with bhp points and save out to first column of resp. array
    well_name = kwargs["well_name"]
    for i, bhp in enumerate(grid):
        w_pred = rate_model({well_name: bhp}, history, *kwargs[SIM_ARGS])[well_name]
        data[i, 0] = get_value_from_pred(w_pred, kind)

    return data

def calculate_working_time(
    cur_date: StrOrTimeStamp,
    last_date: Optional[StrOrTimeStamp]=None,
    factor: Optional[float]=0.95
    ) -> float:
    """Calculate working time of the well in days in a given period.

    Parameters
    ----------
    cur_date : StrOrTimeStamp
        first date after given period
    last_date : Optional[StrOrTimeStamp], optional
        first date of the given period, by default None
    factor : Optional[float], optional
        well working factor (number between 0 and 1), by default 0.95

    Returns
    -------
    float
        total working time in days of the well in selected time range
    """
    cur_date = pd.to_datetime(cur_date)
    if last_date is None:   
        last_date = cur_date + pd.DateOffset(
                months=-1
            )
    num_days = (pd.to_datetime(cur_date) - pd.to_datetime(last_date)).days
    total_days = num_days * factor
    return total_days

def get_last_date_from_df(
        df: pd.DataFrame
    ) -> pd.Timestamp:
    
    return df[DATE_COL].max()