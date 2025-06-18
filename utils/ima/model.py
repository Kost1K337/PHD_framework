"""IMA Forecasting module."""
import time
import numpy as np
import pandas as pd

from astropy import units as u

if __name__ == 'core_engine.ima.model':
    from mipt_solver.nodal.inflow_curves import Curve
elif __name__ == 'ima.model':
    from mipt_solver.nodal.inflow_curves import Curve

from .const import *  # pylint: disable=wildcard-import,unused-wildcard-import
from .ima_utils import *  # pylint: disable=wildcard-import,unused-wildcard-import
from .df_utils import *  # pylint: disable=wildcard-import,unused-wildcard-import
from .load_utils import *  # pylint: disable=wildcard-import,unused-wildcard-import
from .plot_utils import *  # pylint: disable=wildcard-import,unused-wildcard-import
from ._typing import OptionalPath


__ALL__ = ["IMAForecast"]


class IMAForecast:  # pylint: disable=too-many-instance-attributes
    """Class creating object-calculator predicting IMA forecast."""

    def __init__(
        self,
        rate_model: Any,
        network: Any,
        net_solver: Any,
        cfg_path: OptionalPath,
    ) -> None:
        self.rate_model = rate_model
        self.network = network
        self.net_solver = net_solver
        self.cfg_path = Path(cfg_path)
        self.errors = None
        self.total_time = 0
        self.time_per_iter = None
        self.cfg = None
        self.guess = None
        self.iter_num = None
        self.chokes_0 = None
        self.plot_kwargs = None
        self.history = None
        self.full_data = None
        self.wparam = None
        self.skpress = None
        self.bhp_lim = None
        self.horizon = None
        self.raise_errors = False

    def prepare_forecast(self):
        """Prepare all metaparameters taken from config."""
        # load config prototype
        self.load_cfg(self.cfg_path)

        # calculate number of years and months
        self.n_years = np.floor(self.cfg[N_STEPS] / 12).astype(int)
        self.n_months = self.cfg[N_STEPS] - self.n_years * 12
        self.well_names = self.network.wellnames
        self.horizon = self.cfg[N_STEPS]

        ## determine grid density for curve sampling
        self.bhp_dens = self.cfg.get(BHP_GRID_DENSITY, DEF_BHP_GRID_DENSITY)

        # EXPERIMENT CONSTANTS
        self.cfg[EXP_DATE] = get_today()
        self.network_dump_path = self.cfg[NETWORK_DUMP_FORMAT].format(**self.cfg)
        self.start_date = pd.to_datetime(self.cfg[START_DATE])
        self.VAL_KWARGS = {
            LAST_BHP: {well: self.cfg[LAST_BHP.upper()] for well in self.well_names},
            LAST_WCT: {well: 0 for well in self.well_names},
            "sim_args": (0,),
            "interp_kwargs": {"fill_value": "extrapolate"}
        }
        
        self.history = pd.DataFrame()

        if TRUE_DATA_PATH in self.cfg:
            self.full_data, self.history = load_data(self.cfg_path.parent.joinpath('historical_data') / self.cfg[TRUE_DATA_PATH],
                                                     self.start_date,
                                                     self.cfg)

        if SCHEDULE_PATH in self.cfg:
            self.skpress = get_sk_press_dataframe(
                self.cfg_path.parent / self.cfg[SCHEDULE_PATH], 
                sk_name=self.network.sink.name)
            
        if SCHEDULE_PATH_EXCEL in self.cfg:
            self.skpress = get_sk_press_dataframe_from_excel(self.cfg_path.parent.joinpath('historical_data') / self.cfg[SCHEDULE_PATH_EXCEL])

        self.history4pred = deepcopy(self.history)
        self.curves_date = {
            IPR: {},
            WCT_COL: {},
            GOR_COL: {},
            QOIL_COL: {},
            QGAS_COL: {},
            QWAT_COL: {},
        }
        self.wells_to_protract = []
        self.total_time = 0
        self.time_per_iter = None
        self.guess = None
        self.plot_kwargs = None

        # save de facto config
        save_cfg(self.cfg, name_format="{EXP_NAME}_{EXP_DATE}.json")
    
        # self.guess = generate_first_guess(self.well_names, self.history)
        self.chokes_0 = None
        self.errors = pd.DataFrame([], columns=[LOSS])
        set_rate_model_curves(self.network.nodes, self.rate_model, **self.VAL_KWARGS) # NOTE: инициализация кривых на первом шаге

    def forecast_step(self, to_save: bool = False, to_display: bool = False):
        "Make 1 IMA forecast step."

        last_date = pd.to_datetime(self.start_date) + pd.DateOffset(
            months=self.iter_num - 1
        )
        cur_date = pd.to_datetime(self.start_date) + pd.DateOffset(months=self.iter_num)
        cur_date_format = cur_date.strftime("%Y-%m-%d")
        print(f"{STEP} {self.iter_num+1:04d} {DATE}: {cur_date_format}")

        ## update curves parameters
        #update_dca_curves_parameters(self.network, cur_date, last_date)  # TODO: переписать на абстрактное обновление параметров   
        ## full injection for the next time step
        # val_kw = deepcopy(self.VAL_KWARGS)

        ## get last value of pressure, water cut, gor and liquid rates
        # last_bhps = get_last_values(
        #     self.well_names, self.history, last_date, value=WBHP_COL, **val_kw
        # )
        # last_wcts = get_last_values(
        #     self.well_names, self.history, last_date, value=WCT_COL, **val_kw
        # )
        # last_gors = get_last_values(
        #     self.well_names, self.history, last_date, value=GOR_COL, **val_kw
        # )
        # last_qliqs = get_last_values(self.well_names, self.history, last_date, value=QLIQ_COL, **val_kw)

        ## NOTE: not used in nodal-based solution
        # rate_dict = get_rate_dict_from_last(self.well_names, last_qliqs, last_wcts, last_gors)


        # Additional keyword arguments: Sink pressure and where

        if self.skpress is not None:
            skpress = (self.skpress.loc[last_date][PRES_COL] * u.bar).to(u.Pa)
            self.network.sink.p = skpress

        ## generate bounds for optimizer
        # NOTE: bounds
        # net_bounds = generate_network_bounds_mipt(net,
        #                                     mode=self.cfg[RATE_BOUNDS],
        #                                     last_qliqs=last_qliqs,
        #                                     #  last_bhps=last_bhps,
        #                                     ipr_dict=ipr_dict,
        #                                     window=self.cfg[PERCENT_BOUNDS])

        ## initial guess
        if self.cfg[INITIAL_GUESS] != "prev":
            self.guess = None
        # break

        #reservoir_pressures = {}
        # for w, curve in ipr_curves_dict.items():
        #     reservoir_pressures[w] = curve.func(0)
        for node in self.network.nodes:
            if node.ntype == NodeType.SOURCE:
                node.p_reservoir = node.ipr_curve.func(0)

        #self.network.update_well_respres(reservoir_pressures) # TODO: !!!!НЕОБХОДИМО ДОБАВИТЬ ДИНАМИКУ ПЛАСТОВОГО ДАВЛЕНИЯ!!!!

        # fill_wellnodes(self.network, last_gors=last_gors, last_wcts=last_wcts)
        self.net_solver, best_sol, best_err = solve_network_mipt(
            self.net_solver,
            maxiter=self.cfg[MAX_SOLVER_ITER],
            # guess_0=self.guess,
            n_trials=self.cfg[NET_TRIALS],
            n_retrials=self.cfg[NET_RETRIALS],
            n_errors=self.cfg[NET_ERRORS],
            max_error=self.cfg[NET_MAX_ERROR],
            min_error=self.cfg[NET_MIN_ERROR],
            raise_errors=self.raise_errors
        )

        self.errors.loc[cur_date_format] = best_err
        print(f'\n########################################################################')
        print(f"Error at best solution: {best_err:.8f}")
        print(f'########################################################################\n')

        if to_save:
            dump_network_table(
                df=self.network.df,
                path=self.cfg[NETWORK_DUMP_FORMAT],
                sheet_name=str(self.iter_num),
            )

        ## previous rates and autochoke pressure differences for the next step
        self.guess = best_sol[: len(self.network.wellnames) + 1]

        # Note: chokes are not propertly supported in `nodal` by now
        # chokes_0 =  best_sol[len(self.network.wellnames):len(self.network.autochokes)+1]

        net_rates = {
            wn: self.network.df.loc[wn].to_dict() for wn in self.network.wellnames
        }

        ## form dataframes
        pred_df = predict_rates(
            self.well_names,
            cur_date,
            # self.rate_model,
            net_rates,
            # self.history,
            mode=self.cfg[PRED_MODE],
            date_to_protract=last_date,
        )
        # protract well data for current date if wells in `self.wells_to_protract`
        if self.wells_to_protract:
            pred_df = protract_dates(
                pred_df, self.history, self.wells_to_protract, last_date
            )
            
        ## update borehole pressure and current rate parameters for DCA
        set_dca_rate_pressure(self.network, net_rates) 

        ## update history dataframe
        self.history = pd.concat(
            (self.history, pred_df), ignore_index=True
        ).reset_index(drop=True)

    def forecast(
        self, test: bool = False, to_display: bool = False, to_save: bool = False
    ) -> None:
        """Make full forecast.

        Parameters
        ----------
        test : bool, optional
            make single test run, by default False
        to_display : bool, optional
            display tables and figures during the forecast (slows down calculations), by default False
        to_save : bool, optional
            dump tables and figures during the run (slows down calculation), by default False
        """
        ## Create EXCEL file for network table dump
        
        if to_save:
            prepare_network_table_dump(self.network_dump_path)
        
        time_0 = time.time()

        for self.iter_num in range(self.horizon):
            self.forecast_step(to_display=to_display, to_save=to_save)
            if test:
                break

        time_1 = time.time()

        self.total_time, self.time_per_iter = get_delta_time(
            time_0, time_1, self.horizon if not test else 1
        )
        print(
            f"{self.total_time.seconds:.2f} seconds\n{self.time_per_iter.seconds:.2f} sec / iter"
        )

        if to_save:
            self.dump_curves()

    def dump_curves(self):
        """Dump IPR (GOR, WCT) data (after forecast.)"""
        if self.iter_num is None:
            return
        dump_curves_dict(
            self.curves_date,
            name=f"{self.cfg[EXP_NAME]}_{self.cfg[EXP_DATE]}_ipr_bound_curves",
            frmt="json",
        )

    def dump_solver_errors(self, frmt: str) -> None:
        """Dump solver errors to file"""
        frmt_dict = {
            "exp_name": self.cfg[EXP_NAME],
            "exp_date": self.cfg[EXP_DATE],
            "column": LOSS,
        }
        frmt = "{exp_name}_{exp_date}_{column}.csv"
        self.errors.to_csv(frmt.format(**frmt_dict))

    def load_cfg(self, path: OptionalPath = None) -> None:
        """Load IMA config"""
        if not path:
            path = self.cfg_path
        self.cfg = load_cfg(name=path)
        print(self.cfg)
        self.wparam = load_wparam(path.parent / self.cfg['WPARAM_PATH'])
        return self

    def update_cfg(self) -> None:
        """Update IMA config"""
        if self.total_time:
            self.cfg["TOTAL_TIME"] = self.total_time
        if self.time_per_iter:
            self.cfg["TIME_PER_STEP"] = self.time_per_iter
        self.cfg[EXP_DATE] = get_today()
        return self

    def dump_cfg(self, **kwargs) -> None:
        """Dump IMA config"""
        if not kwargs:
            kwargs = {"name_format": "{EXP_NAME}_{EXP_DATE}.json"}

        save_cfg(self.cfg, **kwargs)
