"""Constants used in solver."""

DATE_COL = "date"
WELL_COL = "well"
INJT_COL = "injt"
PRES_COL = "pres"
ALQ_COL = "alq"
QOIL_COL = "qoil"
QWAT_COL = "qwat"
QGAS_COL = "qgas"
QLIQ_COL = "qliq"
WBHP_COL = "bhp"
WCT_COL = "wct"
GOR_COL = "gor"
MODL_COL = "model"
SCEN_COL = "scenario"

INJECTOR = "injector"
TARGET_WELLS = "TARGET_WELLS"
METHOD = "method"
DATA = "data"
IPR = "ipr"
XCOL = "xcol"
YCOL = "ycol"
FUNC = "func"
NET = "_net"
RE = "_r.e."
AE = "_a.e."

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

## IMA constants
DATE = "date"
PRED = "pred"
LOSS = "loss"
HIST = "history"
EXP_DATE = "EXP_DATE"
EXP_NAME = "EXP_NAME"
RATE_BOUNDS = "rate_bounds"
PERCENT_BOUNDS = "percent_bounds"
NET_TRIALS = "NET_TRIALS"
NET_RETRIALS = "NET_RETRIALS"
NET_ERRORS = "NET_ERRORS"
NET_MAX_ERROR = "NET_MAX_ERROR"
NET_MIN_ERROR = "NET_MIN_ERROR"
MAX_SOLVER_ITER = "MAX_ITER"
SLSQP = "SLSQP"
INITIAL_GUESS = "INITIAL_GUESS"
PRED_MODE = "PRED_MODE"
SAMPLING_MODE = "sampling_mode"
N_STEPS = "N_STEPS"
BHP_GRID_DENSITY = "bhp_grid_density"
NETWORK_DUMP_FORMAT = "NETWORK_DUMP_FORMAT"
START_DATE = "START_DATE"
LAST_BHP = "last_bhp"
LAST_WCT = "last_wct"
LAST_GOR = "last_gor"
SIM_ARGS = "sim_args"
STEP = "step"
DETACH = "DETACH"
DATES_TO_FILTER = "dates_to_filter"
SCHEDULE_PATH = "schedule_path"
SCHEDULE_PATH_EXCEL = "schedule_path_excel"
TRUE_DATA_PATH = "true_data_path"
WPARAM_PATH = "wparam_path"
WORK_FACTOR = 0.95

REGEX_SINK_PRESS = r"DATES\s+{}.+?'?{}'?\s+([\d\.]+?)\s*?\/.+?(?=DATES)"
REGEX_SINK_DATE_PRESS = r"DATES\s+?(\d{{2}} +(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC) +\d{{4}}).+?'{}' +([+-]?\d+(?:\.\d+)?(?:[eE][+-]\d+)?) */.+?(?=DATES)"

DEF_BHP_GRID_DENSITY = 1
EPS = 1e-15
DEFAULT_CFG = "mvp_default_config.json"
DEF_PRES_BOUNDS = (0, 1000)
DEF_RATE_BOUNDS = (0, 1000)
DEF_PERCENT_BOUNDS = (-5, 5)
DEF_CHOKE_BOUNDS = (0, 100)
DEF_SAMPLING_WINDOW = (-5, 5)
DEF_FIGSIZE = (21, 3)
DEF_IPR_FORM = "linear"
MAX_Q = 200

UNITS = {WBHP_COL: "bar", QLIQ_COL: "m^3/d", WCT_COL: "m^3/m^3", GOR_COL: "m^3/m^3"}

CURVE_KIND_2_COL_NAME = {
    "oil": QOIL_COL,
    "wat": QWAT_COL,
    "gas": QGAS_COL,
    "gor": GOR_COL,
    "wct": WCT_COL,
    "ipr": QLIQ_COL,
    "bhp": PRES_COL
}