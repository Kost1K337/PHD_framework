"""Utility function for the work of integrated model"""

from copy import deepcopy
from typing import Any, List, Sequence, Union
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

## matplotlib
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import seaborn as sns

if __name__ == 'core_engine.ima.plot_utils':
    from mipt_solver.nodal import custom_units as cu
elif __name__ == 'ima.plot_utils':
    from mipt_solver.nodal import custom_units as cu
from .const import *  # pylint: disable=wildcard-import, unused-wildcard-import


def find_microPI(
    data: pd.DataFrame, sim: Any, last_bhps: dict, *sim_args, dp: float = 1e-1
):
    """IGNORE: function to estimate PI from historical data."""
    pis = {}
    for w in sim.well_list:
        lb = last_bhps[w] - dp / 2
        qs = []
        for _ in range(2):
            a = sim({w: lb}, data, *sim_args)[w]
            qwat = a[QWAT_COL]
            qoil = a[QOIL_COL]
            qliq = qwat + qoil
            qs.append(qliq)
            lb += dp / 2

        pis[w] = (qs[0] - qs[1]) / dp

    return pis


def plot_microPI(
    history: pd.DataFrame,
    well_names: List[str],
    last_bhps: dict,
    sim_args: tuple,
    log: bool = False,
) -> None:
    """Estimate PI from finite difference estimation.

    Parameters
    ----------
    history : pd.DataFrame
        dataframe
    well_names : List[str]
        _description_
    last_bhps : dict
        _description_
    sim_args : tuple
        _description_
    log : bool, optional
        _description_, by default False
    """
    fr = {w: {DATA: [], "mode": None} for w in well_names}
    dps = np.linspace(-3, 1, num=100)
    dps = 10**dps
    for dp in dps:
        d = find_microPI(history, last_bhps, *sim_args, dp=dp)

        fr[w][DATA].extend(list(d.values()))

    for w in fr:
        p = plt.plot(dps, fr[w][DATA], label=f"Well {w} PI")
        plt.xlabel("$\delta BHP, bar$")
        plt.ylabel("$J, m^3/d/bar$")
        mode = np.median(fr[w][DATA])
        plt.axhline(
            mode, color=p[0].get_color(), linestyle="--", label=f"{w} PI median"
        )

    plt.legend()
    plt.title("Productivity Index from finite difference estimation")
    if log:
        plt.xscale("log")
        plt.yscale("log")


def plot_curves_vs_points(
    curve_dict: dict,
    dates: List[Union[pd.DataFrame, str]],
    start_date: Union[pd.DataFrame, str],
    true: pd.DataFrame,
    pred: pd.DataFrame,
    col: str,
    **kwargs,
) -> None:
    """Plot IPR (GOR, WCT) curves and points from actual data

    Parameters
    ----------
    curve_dict : dict
        dictionary containing curves for every well
    dates : List[Union[pd.DataFrame,str]]
        Dates where to split time ranges
    start_date : Union[pd.DataFrame,str]
        _description_
    true : pd.DataFrame
        dataframe containing ground truth forecast
    pred : pd.DataFrame
        dataframe containing predicted forecast
    col : str
        column name to plot against BHP (`QLIQ_COL`, `WCT_COL`, `GOR_COL`)
    """
    xslack = kwargs.get("xslack", 5)
    yslack = kwargs.get("yslack", 5)
    xlabel_ = f"{col.upper()}, ${cu.UNITS[col]}$"
    xlabel = kwargs.get("xlabel", xlabel_)
    ylabel = kwargs.get("ylabel", "BHP, bar")
    log = kwargs.get("log", 0)
    suptitle = kwargs.get("suptitle", "tNav data vs. IPR")
    ncols = kwargs.get("ncols", 6)
    nrows = kwargs.get("nrows", len(dates) - 1)

    fig = plt.figure(constrained_layout=True, figsize=(4 * ncols, 4 * nrows), dpi=300)
    subfigs = fig.subfigures(nrows=nrows, ncols=1)
    true_data = true[(true.date >= start_date) & (true.date <= pred.date.max())]

    for rr in range(nrows):
        from_date = dates[rr]
        till_date = dates[rr + 1]
        subfigs[rr].suptitle(
            f"С {from_date.strftime('%Y-%m-%d')} до {till_date.strftime('%Y-%m-%d')}"
        )
        axs = subfigs[rr].subplots(ncols=ncols, nrows=1)

        for i, (well, wdf) in enumerate(true_data.groupby(WELL_COL)):
            wdff = wdf[(wdf[DATE_COL] <= till_date) & (wdf[DATE_COL] >= from_date)]
            qliq, wbhp = wdff[col], wdff[WBHP_COL]

            wdff_net = pred[
                (pred[DATE_COL] <= till_date)
                & (pred[DATE_COL] >= from_date)
                & (pred[WELL_COL] == well)
            ]
            qliq_net, wbhp_net = wdff_net[col], wdff_net[WBHP_COL]

            color = cm.rainbow(np.linspace(0, 1, num=len(qliq)))
            j = 0

            for k, v in curve_dict.items():
                if pd.to_datetime(k) > till_date:
                    break
                if pd.to_datetime(k) > from_date:
                    if DATA in v[well]:
                        x_grid, y_grid = v[well][DATA][:, 0], v[well][DATA][:, 1]
                    else:
                        pres = wdf.loc[
                            wdf[DATE_COL] == pd.to_datetime(k), PRES_COL
                        ].item()
                        x_grid = np.linspace(0, pres + xslack, num=100)
                        y_grid = v[well][FUNC](x_grid)

                    # IPR or whatever curve
                    axs[i].plot(x_grid, y_grid, c=color[j], alpha=0.45)
                    j += 1

            axs[i].scatter(qliq, wbhp, c=color, s=40)
            axs[i].scatter(qliq_net, wbhp_net, c=color, marker="x", s=40)
            qliq_min = np.min(qliq_net.to_list() + qliq.to_list())
            wbhp_min = np.min(wbhp_net.to_list() + wbhp.to_list())
            qliq_max = np.max(qliq_net.to_list() + qliq.to_list())
            wbhp_max = np.max(wbhp_net.to_list() + wbhp.to_list())
            axs[i].set_xlim([qliq_min - xslack, qliq_max + xslack])
            axs[i].set_ylim([wbhp_min - yslack, wbhp_max + yslack])

            if log:
                axs[i].set_yscale("log")
                axs[i].set_xscale("log")
            axs[i].set_xlabel(xlabel)
            axs[i].set_title(well)
        axs[0].set_ylabel(ylabel)

    fig.suptitle(suptitle)
    plt.show()


def plot_cumulative(
    cum_true: pd.DataFrame,
    cum_pred: pd.DataFrame,
    scale: Union[Sequence, float] = 100,
    deviance: Union[Sequence, float] = 5,
    **plt_kwargs,
) -> None:
    """Plot cumulative data for QLIQ, QOIL, QGAS, QWAT.

    Parameters
    ----------
    cum_true : pd.DataFrame
        dataframe with ground truth cumulative rates, generated by `calculate_cumulative`
    cum_pred : pd.DataFrame
        dataframe with predicted cumulative rates, generated by `calculate_cumulative`
    scale : Union[Sequence,float], optional
        scale factor for axis, by default 100
    deviance : Union[Sequence,float], optional
        corridor size around ground truth curve in percents, by default 5

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """
    vals = [QLIQ_COL, QOIL_COL, QGAS_COL, QWAT_COL]
    plt_kwargs["figsize"] = plt_kwargs.get("figsize", (20, 5))
    plt_kwargs["dpi"] = plt_kwargs.get("dpi", 100)
    _, axs = plt.subplots(ncols=4, **plt_kwargs)
    font = {"family": "normal", "weight": "regular", "size": 10}
    matplotlib.rc("font", **font)

    for i, val in enumerate(vals):
        if isinstance(scale, (list, tuple)):
            scl = scale[i]
        elif isinstance(scale, (float, int)):
            scl = scale
        else:
            raise ValueError("`scale` should be a number or list or a tuple")

        if isinstance(deviance, (list, tuple)):
            dev = deviance[i]
        elif isinstance(deviance, (float, int)):
            dev = deviance
        else:
            raise ValueError("`deviance` should be a number or list or a tuple")

        axs[i].plot(
            range(len(cum_true.date)),
            cum_true[val] / scl,
            label="Ground Truth",
            color="g",
        )
        axs[i].plot(
            range(len(cum_pred.date)),
            cum_pred[val] / scl,
            label="Prediction",
            color="r",
        )
        axs[i].fill_between(
            range(len(cum_pred.date)),
            y1=(1 - dev / 100) * (cum_true[val] / scl),
            y2=(1 + dev / 100) * (cum_true[val] / scl),
            label=f"{dev}% deviation",
            alpha=0.15,
            color="g",
        )
        axs[i].set_xlabel("months")
        axs[i].set_ylabel(f"Cum. {val.upper()} Production, ${scl} m^3$")
        axs[i].legend()
    plt.tight_layout()
    plt.show()


def plot_errors(comp_melt: pd.DataFrame, save_fig: Union[str, Path]) -> None:
    sns.set(font_scale=1.5)
    g = sns.relplot(
        data=comp_melt,
        x=DATE_COL,
        y="value",
        col=WELL_COL,
        hue="variable",
        row="error_type",
        kind="line",
        facet_kws=dict(sharey=False),
    )
    g.set_xticklabels(rotation=30)
    g.axes[0, 0].set_ylim([-60, 40])
    for i in range(6):
        g.axes[1, i].set(yscale="log")
    plt.savefig(save_fig)


def plot_profiles(reldata: pd.DataFrame, save_fig: Union[str, Path]) -> None:
    """Plot predicted profile against ground_truth"""
    sns.set(font_scale=1.5)
    
    reldata.date = pd.to_datetime(reldata.date)
    g = sns.relplot(
        data=reldata,
        kind="line",
        y="value",
        x=DATE_COL,
        style="label",
        col=WELL_COL,
        hue="var",
        row="var",
        facet_kws=dict(sharey=False),
        linewidth=2,
    )

    for i in range(6):
        g.axes[0, i].set(yscale="log")

    g.set_xticklabels(rotation=30)
    plt.savefig(save_fig)
    plt.show()


def fit_ipr(
    x: Union[list, np.array], y: Union[list, np.array], form: str = "linear"
) -> tuple:
    """Take np.array's X (usually, liquid rate) and y (usually, BHP)
    Fit linear IPR form and return PI (J), reservoir pressure (Pres) and predicted values (BHP)

    Parameters
    ----------
    x : Union[list,np.array]
        1D array of QLIQ
    y : Union[list,np.array]
        1D array of BHP
    form : str, optional
        form of a curve, by default "linear"

    Returns
    -------
    tuple
        tuple of PI, Pres and predicted BHP

    Raises
    ------
    NotImplementedError
        Raised when a form of curve given by `form` is not supported.
    """
    if form not in ("linear",):
        raise NotImplementedError(f"`form` value {form} is not supported yet.")
    l_r = LinearRegression()
    l_r.fit(x, y)
    j = (
        -1 / l_r.coef_.item()
    )  # Productivity Index estimate: units: [j] = [q] / [P] = (m^3 / d) / bar
    p_r = l_r.intercept_.item()  # Reservoir pressure: units: [p_r] = bar
    y_pred = l_r.predict(x)  # predicted BHPs
    return (j, p_r, y_pred)  # return PI, Pr and (usually, BHP) estimate


def plot_curves(
    dct: dict,
    xcol: str = QLIQ_COL,
    ycol: str = WBHP_COL,
    save: bool = False,
    plot: bool = True,
    start_0: bool = False,
    **plt_kwargs,
):
    """Plot curves (IPR, WCT, GOR) containing in `dct`

    Parameters
    ----------
    dct : dict
        dict containing curve data for every well in question
    xcol : str, optional
        abscissa, by default QLIQ_COL
    ycol : str, optional
        ordinate, by default WBHP_COL
    save : bool, optional
        whether to save plots to file, by default False
    plot : bool, optional
        whether to demonstrate the plot, by default True
    start_0 : bool, optional
        whether to limit the axes by origin (0,0), by default False

    Raises
    ------
    ValueError
        Raised when `xcol` not supported.
    ValueError
        Raised when `ycol` not supported.
    """
    ## Copy to protect
    dict_ = deepcopy(dct)

    ## Make sure column values are supported.
    if (
        xcol not in (QLIQ_COL, WCT_COL, GOR_COL)
        or ycol not in (QLIQ_COL, WCT_COL, GOR_COL, WBHP_COL)
        or ycol == xcol
    ):
        raise ValueError("`xcol` or `ycol` value is not supported.")

    plt_kwargs[XCOL] = xcol
    last_bhps = plt_kwargs.get("last_bhps")

    fontp = FontProperties()
    fontp.set_size("xx-small")
    fig, axs = plt.subplots(
        ncols=len(dict_), figsize=plt_kwargs.get("figsize", DEF_FIGSIZE)
    )

    for i, well in enumerate(dict_.keys()):
        data = dict_[well]
        if DATA in data:
            x, y = data[DATA][:, 0].reshape(-1, 1), data[DATA][:, 1].reshape(-1, 1)
        else:
            grids = plt_kwargs.get("grids", np.linspace(0, MAX_Q))
            y = grids[well].reshape(-1, 1)
            x = data[FUNC](y, **{k: v for k, v in data.items() if k != FUNC})
        axs[i].plot(x, y, label=plt_kwargs.get("label", "Raw data"))

        if xcol in (QLIQ_COL,):
            out = fit_ipr(x, y, form=plt_kwargs.get("ipr_analytic_form", DEF_IPR_FORM))
            coef, y_pred = out[:-1], out[-1]
            axs[i].plot(x, y_pred, label=f"LR (J={coef[0]:.2f}, P={coef[1]:.2f})")

        axs[i].axhline(last_bhps[well], linestyle="--", color="g", label="last bhp")
        axs[i].set_xlabel(f"${xcol}, {cu.UNITS[xcol]}$")
        axs[i].set_ylabel(f"${ycol}, {cu.UNITS[ycol]}$")
        axs[i].set_title(well)
        if start_0:
            axs[i].set_xlim(left=0)
            axs[i].set_ylim(bottom=0)
        axs[i].legend(prop=fontp)

    fig.suptitle(xcol)
    plt.tight_layout()
    if save:
        plt.savefig(plt_kwargs["filename_format"].format(**plt_kwargs))
    if plot:
        plt.show()
    plt.close()


def check_VFP(well_names: list, df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(ncols=6, figsize=(24, 4))
    for i, well_name in enumerate(well_names):
        cols = [DATE_COL, METHOD, well_name]
        wdf = df[cols]
        x = wdf[wdf.method == "tNav"][well_name]
        y = wdf[wdf.method == "net"][well_name]
        axs[i].scatter(x, y)
        axs[i].plot(x, x, alpha=0.5)
        axs[i].set_xlabel("True BHP, Bar")
        axs[i].set_ylabel("Pred BHP, Bar")
        axs[i].legend()

        fig.suptitle("VFP")
