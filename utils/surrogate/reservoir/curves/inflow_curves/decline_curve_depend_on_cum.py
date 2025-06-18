from .curve import Curve
import astropy.units as u
from .. import custom_units as cu
import scipy as sc
import numpy as np
if __name__ == 'core_engine.mipt_solver.nodal.inflow_curves.decline_curve_depend_on_cum':
    from core_engine.ima.curve_utils import * # pylint: disable=wildcard-import
elif __name__ == 'mipt_solver.nodal.inflow_curves.decline_curve_depend_on_cum':
    from ima.curve_utils import *  # pylint: disable=wildcard-import


class DeclineCurveDependedOnCum(Curve):     # for GOR and WCUT we do noy initiazie curves 
    def __init__(self, **kwargs):
        super().__init__(params={}, kind=kwargs['kind'])
        self.PI = None  # productivity index
        self.Qopen = None # reservoir open flow debit
        self.cum = 0 * u.m ** 3  # cum fluid
        self.Q_prev = kwargs.get("Q_prev", 1 * cu.CUB) # debit on previous step
        self.Pbhp_prev = kwargs.get("Pbhp_prev", None) # borehole pressure on previous step 
        self.cum_ab_table = kwargs.get("cum_ab_table", None) #{"cum": [], "a" : [], "b" : []} # table of decline curve coefficients for history cum statistics                   
        self.validate()
        self.update_ipr()

    def validate(self):

        a_unit = self.cum_ab_table["a"].unit
        b_unit = self.cum_ab_table["b"].unit
        cum_unit = self.cum_ab_table["cum"].unit

        cum_values = np.array([quant.to(cum_unit).value for quant in self.cum_ab_table["cum"]])
        a_values = np.array([quant.to(a_unit).value for quant in self.cum_ab_table["a"]])
        b_values = np.array([quant.to(b_unit).value for quant in self.cum_ab_table["b"]])

        pi_left = a_values * cum_values + b_values
        if (pi_left < 0).sum() != 0:
            raise ValueError("Decline curve validation failed")

        pi_right = a_values[:-1] * np.roll(cum_values, -1)[:-1] + b_values[:-1]
        if (pi_right < 0).sum() != 0:
            raise ValueError("Decline curve validation failed")

    def set_cum_ab_table(self, table): # cum table format {"cum": [],"a" : [],"b" : []}
        self.cum_ab_table = table
        
    def update_cum(self, days_delta: u.Quantity): # update cum in the end of 1 iteration
        self.cum = self.cum + self.Q_prev * days_delta
        
    def update_ipr(self): # method for update IPR curv in every step
        a, b = self.culc_ab(self.cum) # get a,b coefficients in equation PI = a * cum + b
        cum = self.cum
        max_cum = max(self.cum_ab_table["cum"])
        if cum > max_cum:
            cum = max_cum
        self.PI = a * cum + b  # PI updated
        self.Qopen = self.Q_prev + self.PI * self.Pbhp_prev  # Qopen updated
        
    def culc_ab(self, cum): # method for update PI a,b coefficients in every step
        a_unit = self.cum_ab_table["a"].unit
        b_unit = self.cum_ab_table["b"].unit
        cum_unit = self.cum_ab_table["cum"].unit

        cum_values = [quant.to(cum_unit).value for quant in self.cum_ab_table["cum"]]
        a_values = [quant.to(a_unit).value for quant in self.cum_ab_table["a"]]
        b_values = [quant.to(b_unit).value for quant in self.cum_ab_table["b"]]

        # interpolate a, b depended on cum
        a_inter = sc.interpolate.interp1d(x = cum_values, y = a_values, kind = 'nearest', fill_value="extrapolate")
        b_inter = sc.interpolate.interp1d(x = cum_values, y = b_values, kind = 'nearest',  fill_value="extrapolate")
        return a_inter(cum) * a_unit, b_inter(cum) * b_unit
        
    def direct(self, Pbhp): # calculate debit from actual IPR curve
        Q = -self.PI * Pbhp + self.Qopen
        if Q < 0:
             Q = 0.1
        return Q
    
    def inverse(self, Q):
        Pbhp = (self.Qopen - Q) / self.PI 
        return Pbhp
    
    def update(self, well_history, **kwargs):
        
        # determine column to take values from
        value_col = CURVE_KIND_2_COL_NAME[self.kind]
        bhp_col = CURVE_KIND_2_COL_NAME["bhp"]
        # extract last rates from history
        last_rate = get_last_well_value(history=well_history,
                                        value=bhp_col,
                                        **kwargs
                                        )
        last_bhp = get_last_well_value(history=well_history,
                                       value=value_col,
                                       **kwargs
                                       )
        
        # update rate and pressure
        self.Q_prev = last_rate * cu.UNITS[value_col]
        self.Pbhp_prev = last_bhp * cu.UNITS[bhp_col]
        
        # calculate total working_time incl. well working factor
        cur_date_df = get_last_date_from_df(well_history)
        cur_date = kwargs.get("cur_date", cur_date_df)
        last_date = kwargs.get("last_date", None)
        time_factor = kwargs.get("time_factor", WORK_FACTOR)
        working_days_with_factor = calculate_working_time(cur_date,
                                                          last_date,
                                                          time_factor
                                                          )
        working_days_with_factor_dim = working_days_with_factor * u.day
        # update cumulative rate
        self.update_cum(working_days_with_factor_dim)
        # update parameters of the curve
        self.update_ipr()

class IprCurveDependedOnCum(DeclineCurveDependedOnCum):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, kind='ipr')

    def direct(self, Q):
        return super().inverse(Q)
        
    def inverse(self, Pbhp):
        return super().direct(Pbhp)
