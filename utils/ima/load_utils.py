"""Utility function loading files for IMA"""

from typing import List 

## pure utilities
import pickle
import re
import json
from copy import deepcopy
from typing import Iterable, Optional, Tuple, Union
import astropy.units as u
from pathlib import Path

import pandas as pd

from .const import *  # pylint: disable=wildcard-import, unused-wildcard-import
from ._typing import StrOrTimeStamp, OptionalPath


def format_date(date: StrOrTimeStamp, fstring: str = "%d %b %Y") -> str:
    """Format date according to `fstring`"""
    return pd.to_datetime(date).strftime(fstring).upper()


def load_cfg(name: str = DEFAULT_CFG) -> dict:
    """Load IMA config as a dict."""
    with open(name, mode="r", encoding="utf-8") as f:
        return json.load(f)


def save_cfg(dct: dict, **kwargs) -> None:
    """Save dict containing config to a json file.

    Parameters
    ----------
    dct : dict
        Dump IMA config dictionary as json
    """
    if not ("name_format" in kwargs or "name" in kwargs):
        raise ValueError(
            "Either `name_format` or `name` keyword parameter should be provided."
        )
    if "name_format" in kwargs:
        name = kwargs["name_format"].format(**dct)
    elif "name" in kwargs:
        name = kwargs["name"]

    with open(name, mode="w", encoding="utf-8") as f:
        json.dump(dct, f, indent=1)


def load_data(
    true_data_path: Path,
    start_date: StrOrTimeStamp,
    cfg: dict,
    cols_to_drop: list = ["Swat", "Soil", "Sgas", "Rs", MODL_COL, SCEN_COL, INJECTOR],
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the historical data for IMA prediction.

    Parameters
    ----------
    true_data_path: Path
        path to file with true data to compare forecast to
    schedule_path: Path
        file schedule containing changes to network
    start_date : StrOrTimeStamp
        start of the forecast date
    cfg : dict
        IMA predictor config
    cols_to_drop : list, optional
        columns to drop from dataframe, by default ["Swat", "Soil", "Sgas", "Rs", MODL_COL, SCEN_COL, INJECTOR]

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        dataframes of full ground truth data, historical data, dataframe with 1 columns `pres`
    """

    ## fix model id
    models = cfg.get(MODL_COL, 1)
    if isinstance(models, int):
        models = [models]

    ## fix scenario id
    scenarios = cfg.get(SCEN_COL, 0)
    if isinstance(scenarios, int):
        scenarios = [scenarios]
        
    # fix dates to ignore, if none empty list
    dates_to_filter = cfg.get(DATES_TO_FILTER, [])
    
    ## make sure start_date is a date
    start_date = pd.to_datetime(start_date)

    # Load and process dataframe with historical data
    ## load historical data
    full_data = pd.read_csv(true_data_path, parse_dates=[DATE_COL])
    full_data[WELL_COL] = full_data[WELL_COL].astype(str)
    
    ## fix target wells if none, use all the wells
    target_wells = cfg.get(TARGET_WELLS, None)
    if target_wells is None:
        target_wells = full_data[WELL_COL].unique().tolist()

    target_wells = cfg.get(TARGET_WELLS, None)
    if target_wells is None:
        target_wells = full_data[WELL_COL].unique()
        
    ## clean unwanted columns
    if "Unnamed: 0" in full_data:
        del full_data["Unnamed: 0"]
        
    ## select only actual model, scenario, dates and wells
    full_data = full_data[
        ~full_data[DATE_COL].isin(dates_to_filter)
        & full_data[MODL_COL].isin(models)
        & full_data[SCEN_COL].isin(scenarios)
        & full_data[WELL_COL].isin(target_wells)
    ].drop(cols_to_drop, axis=1)

    ## define history as data before `start_date`
    history = full_data.loc[(full_data[DATE_COL] < start_date), :]

    return full_data, history


def get_sk_press_dataframe(
    schedule_path: str, sk_name: Optional[str]
) -> pd.DataFrame:
    """Load and form dataframe  of sink pressures."""
    if sk_name is None:
        return None
    with open(schedule_path, "r", encoding="utf-8") as f:
        sch_text = f.read()
    sp = get_sink_pressure(sch_text, sk_name=sk_name)
    skpress = pd.DataFrame(sp, columns=[DATE_COL, PRES_COL])
    skpress.set_index(DATE_COL, inplace=True)
    # skpress[PRES_COL] = skpress[PRES_COL] * u.bar
    return skpress

class SinkParsingError(Exception):
    """Raised when there is error parsing schedule for sink pressures."""

def get_sk_press_dataframe_from_excel(shedule_path: str) -> pd.DataFrame:
    skpress = pd.read_excel(shedule_path, skiprows=1, header=None)
    skpress.rename(columns={0: DATE_COL, 1: PRES_COL}, inplace=True)
    skpress.set_index(DATE_COL, inplace=True)

    skpress[PRES_COL] = skpress[PRES_COL] * 10

    return skpress

def get_sk_press_dataframe_from_request(skpress_data: List[tuple]) -> pd.DataFrame:
    """Формирует датафрейм skpress из параметра запроса."""
    skpress = pd.DataFrame(skpress_data, columns=[DATE_COL, PRES_COL])
    skpress.rename(columns={0: DATE_COL, 1: PRES_COL}, inplace=True)
    skpress.set_index(DATE_COL, inplace=True)

    skpress[PRES_COL] = skpress[PRES_COL] * 10

    return skpress


def get_sink_pressure(
    text: str, 
    sk_name: str = "SEP"
) -> Union[float, None]:
    """Take sink pressure from schedule file, matching everything from one DATES declaration to another.
    To be deprecated once full blown parser is used with the integrated model forecast.
    """
    # search sink pressure value as value between two `DATES`
    # FIXME: may not be robust, replace by full-blown parser in the final version
    lt = re.findall(
        REGEX_SINK_DATE_PRESS.format(sk_name),
        text,
        flags=re.DOTALL,
    )
    dates_pressures = []
    if lt:
        for match_ in lt:
            try:
                date = pd.to_datetime(match_[0])
                press = float(match_[2])
            
            except Exception as e:
                raise SinkParsingError("Error when parsing date and sink pressure") from e
            dates_pressures.append((date,press))
    return dates_pressures


def dump_curves_dict(
    dct, name: OptionalPath = "curves_dump", frmt: str = "pickle"
) -> None:
    if frmt == "pickle":
        pickle.dump(dct, open(f"{name}.sav", "wb"))
        print("PICKLE DUMPED")
        return

    if frmt == "json":
        json_curves_date = deepcopy(dct)
        for _, v1 in json_curves_date.items():
            for _, v2 in v1.items():
                for _, v3 in v2.items():
                    v3["data"] = v3["data"].tolist()

        json.dump(json_curves_date, open(f"{name}.json", "w"))
        print("JSON DUMPED")
        return

    raise ValueError(f"Unknown format: {frmt}. Choose either `json` or `pickle`")


def prepare_network_table_dump(path: Union[str, Path]) -> None:
    with pd.ExcelWriter(path, mode="w") as writer:
        pd.DataFrame([0], columns=[0]).to_excel(writer, sheet_name="0")


def dump_network_table(
    df: pd.DataFrame, path: Union[str, Path], sheet_name: str
) -> None:
    with pd.ExcelWriter(path, mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet_name)
