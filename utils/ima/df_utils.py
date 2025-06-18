"""Utility function for the work with DataFrames."""
import datetime
from typing import Dict, List, Iterable, Sequence, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .const import *  # pylint: disable=wildcard-import, unused-wildcard-import
from astropy import units as u
if __name__ == 'core_engine.ima.ima_utils':
    from mipt_solver.nodal import custom_units as cu
elif __name__ == 'ima.ima_utils':
    from mipt_solver.nodal import custom_units as cu


def make_comp(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Merge dataframes on 'date' and 'well'  columns and
    create relative deviation column for each pair of duplicated columns.

    Parameters
    ----------
    df1 : pd.DataFrame
        dataframe with IAM prediction (should have `DATE_COL` and `WELL_COL`)
    df2 : pd.DataFrame
        dataframe with simulator (tNav) prediction (should have `DATE_COL` and `WELL_COL`)

    Returns
    -------
    pd.DataFrame
        dataframe with comparison
    """
    comp = df1.merge(df2, on=[WELL_COL, DATE_COL], suffixes=(NET, ""))

    for col in comp.columns:
        if col.endswith(NET):
            col_ = col.split(NET)[0]
            abs_ = comp[col] - comp[col_]
            comp[col_ + RE] = abs_ / (comp[col_] + EPS) * 100

    return comp.set_index([DATE_COL, WELL_COL]).sort_index(axis=1)


def make_short_comp(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    date: Union[str, pd.Timestamp],
    todrop1: Sequence[str] = [
        "Swat",
        "Soil",
        "Sgas",
        "Rs",
        MODL_COL,
        SCEN_COL,
        INJT_COL,
        PRES_COL,
    ],
    todrop2: Sequence[str] = [INJT_COL, PRES_COL],
) -> pd.DataFrame:
    fdf = df1[(df1.date == date) & (df1.date <= df2.date.max())].drop(columns=todrop1)
    fdf[WELL_COL] = fdf[WELL_COL].astype(str)
    pdf = df2[df2.date >= date].drop(columns=todrop2)
    pdf[WELL_COL] = pdf[WELL_COL].astype(str)
    comp = make_comp(pdf, fdf)
    comp[QLIQ_COL] = comp[QWAT_COL] + comp[QOIL_COL]
    comp[QLIQ_COL + NET] = comp[QWAT_COL + NET] + comp[QOIL_COL + NET]
    comp[QLIQ_COL + RE] = (
        (comp[QLIQ_COL + NET] - comp[QLIQ_COL]) / (comp[QLIQ_COL] + EPS) * 100
    )

    return comp


def make_relplot_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df1_ = df1.copy(deep=True).melt([DATE_COL, WELL_COL])
    df2_ = df2.copy(deep=True).melt([DATE_COL, WELL_COL])
    df1_["label"], df2_["label"] = "pred", "true"
    comp = pd.concat((df1_, df2_))

    return comp


def add_comp_cols(
    df: pd.DataFrame, true_col: str = "tNav", pred_col: str = "net"
) -> pd.DataFrame:
    df.loc["r.e"] = (
        (df.loc[pred_col] - df.loc[true_col]) / (df.loc[true_col] + EPS) * 100
    )
    df.loc["a.e"] = df.loc[pred_col] - df.loc[true_col]

    return df


def complete_data_with_cols(data: pd.DataFrame) -> None:
    """Complete dataframe with columns `wct`, `gor`, `qliq` if necessary
    Parameters
    ----------
    data : pd.DataFrame
        dataframe
    """
    if QOIL_COL not in data:
        raise ValueError(f"{QOIL_COL} not found in DataFrame.")

    if QWAT_COL in data:
        data[QLIQ_COL] = data[QWAT_COL] + data[QOIL_COL]
        data[WCT_COL] = data[QWAT_COL] / (data[QLIQ_COL] + EPS)
    if GOR_COL in data:
        data[GOR_COL] = data[QGAS_COL] / (data[QOIL_COL] + EPS)


def calculate_cumulative(
    data: pd.DataFrame, start_date: Union[str, pd.Timestamp]
) -> pd.DataFrame:
    """Calculate cumulative rates across wells."""
    cols = [QOIL_COL, QGAS_COL, QWAT_COL, INJT_COL]
    cum_data = (
        data.groupby(by=DATE_COL)
        .agg(
            {
                WELL_COL: "first",
                DATE_COL: "first",
                PRES_COL: np.mean,
                WBHP_COL: np.mean,
                QOIL_COL: np.sum,
                QGAS_COL: np.sum,
                QWAT_COL: np.sum,
                INJT_COL: np.sum,

            }
        )
        .reset_index(drop=True)
    )

    cum_data = cum_data.drop([WELL_COL], axis=1)
    cum_data = cum_data[cum_data.date >= start_date]
    cum_data[cols] = cum_data[cols].cumsum()
    cum_data[QLIQ_COL] = cum_data[QOIL_COL] + cum_data[QWAT_COL]

    return cum_data


def time_pad_df_with_wells(
    df: pd.DataFrame, freq: str = "MS", **reindex_kwargs
) -> pd.DataFrame:
    """Time pad wells production historical data `df` with with time frequency `freq`
    from minimal date to maximal date in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        original dataframe
    freq : str, optional
        resulting time frequency of data, by default "MS"
    reindex_kwargs : dict, optional
        keywords to pass to reindex method, by default empty dict
    Returns
    -------
    pd.DataFrame
        another dataframe with timepadded data
    """
    all_ts = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq=freq)

    dfs = []
    for well, df_well in df.groupby(WELL_COL):
        df_well = df_well.set_index(DATE_COL).reindex(all_ts, **reindex_kwargs)
        df_well[WELL_COL] = well
        dfs.append(df_well)

    dfs = pd.concat(dfs)
    dfs.index = dfs.index.set_names([DATE_COL])
    return dfs.reset_index()


def time_pad_df(df: pd.DataFrame, freq: str = "MS", **kwargs) -> pd.DataFrame:
    """ime pad `df` with time frequency `freq`
    from minimal date to maximal date in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        original dataframe
    freq : str, optional
        resulting time frequency of data, by default "MS"
    reindex_kwargs : dict, optional
        keywords to pass to reindex method, by default empty dict
    Returns
    -------
    pd.DataFrame
        another dataframe with timepadded data
    """
    all_ts = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq=freq)
    dfs = df.set_index(DATE_COL).reindex(all_ts, **kwargs)
    dfs.index = dfs.index.set_names([DATE_COL])
    return dfs


def protract_dates(
    df: pd.DataFrame,
    history: pd.DataFrame,
    wells_to_protract: Iterable[str],
    last_date: Union[pd.Timestamp, str],
    columns: Iterable = (QOIL_COL, QWAT_COL, QGAS_COL, WBHP_COL, PRES_COL),
) -> pd.DataFrame:
    date_to_protract = last_date
    for well in wells_to_protract:
        wind = df[df[WELL_COL] == well].index
        for val in columns:
            df.loc[wind, val] = history.loc[
                (history[WELL_COL] == well) & (history[DATE_COL] == date_to_protract),
                val,
            ].item()

    return df


def melt_comparison_df(comp: pd.DataFrame) -> pd.DataFrame:
    """Melt the comparison dataframe to plot errors."""
    for val in [WBHP_COL, QGAS_COL, QOIL_COL, QWAT_COL]:
        comp[val+AE] = comp[val+NET] - comp[val]

    compp = comp.reset_index()

    compp = compp[[DATE_COL, WELL_COL, 
                    WBHP_COL+RE, QGAS_COL+RE, QWAT_COL+RE, QOIL_COL+RE, 
                    WBHP_COL+AE, QGAS_COL+AE, QWAT_COL+AE, QOIL_COL+AE]]
    comp_melt = compp.copy(deep=True).melt([DATE_COL, WELL_COL])
    comp_melt["error_type"] = comp_melt["variable"].str.split("_").map(lambda x: x[1])
    comp_melt["variable"] = comp_melt["variable"].str.split("_").map(lambda x: x[0])
    
    return comp_melt

def make_reldata_and_comp(full_data, history, start_date):
    """Create reldata-type dataframe for plotting profiles and comp - dataframe with relative errors
    """
    fdf = full_data[(full_data.date >= start_date) & (full_data.date <= history.date.max())].drop(columns=[PRES_COL, INJT_COL])
#     fdf = full_data[(full_data.date >= start_date) & (full_data.date <= history.date.max())]
    fdf.well = fdf.well.astype(str)
    pdf = history.drop(columns=[INJT_COL, PRES_COL])
    pdf.well = pdf.well.astype(str)

    comp__ = make_comp(pdf, fdf)
    reldata = make_relplot_data(pdf, fdf)
    reldata = reldata.rename(columns={"variable":"var"})
    reldata = reldata.dropna()
    return reldata, comp__
