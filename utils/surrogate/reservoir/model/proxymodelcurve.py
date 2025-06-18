import src.inno_reservoir.model.proxymodelnet as pmn
import src.inno_reservoir.model.curvecontroller as cc
import torch as tc

class ProxyModelNetCurve(pmn.ProxyModelNet): # NOTE: модель пласта
    def __init__(self, well_count, dt, hidden_count = 0, addition_cells = 0, controller: cc.IProxyController = None):
        super().__init__(well_count, dt, hidden_count, addition_cells)
        self.controller = controller

    def set_controller(self, controller: cc.IProxyController):
        self.controller = controller

    def view_internal_prod(self, state: pmn.ProxyModelNet.State):
        self.calc_mobility(state)

        pressure = state.pressure[:,:self.well_count]

        J = self.J * self.J
        state.J_t = tc.where(state.status == 1, J * state.mob_prod_t[:,:self.well_count], J)
        state.J_w = tc.where(state.status == 1, J * state.saturation[:,:self.well_count], J)

        state.production = state.J_t * (pressure - state.bhp_bnd) 

    def view_internal(self, state: pmn.ProxyModelNet.State):
        self.view_internal_prod(state)

        pressure = state.pressure[:,:self.well_count]
        incorrect_status = tc.logical_or(state.production * state.status < 0.0, state.status == 0)

        state.production = tc.where(incorrect_status, 0.0, state.production)
        state.bhp = pressure - state.production / state.J_t

        state.production_w = state.J_w * (pressure - state.bhp)

    def view_internal0(self, state: pmn.ProxyModelNet.State):
        self.view_internal_prod(state)

        pressure = state.pressure[:,:self.well_count]
        state.production = tc.where(state.status == 0, 0.0, state.production)
        state.bhp = pressure - state.production / state.J_t
        state.production_w = state.J_w * (pressure - state.bhp)
        state.J = self.J * self.J
    
    def veiw_weights(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        well_count = self.well_count
        bhp_bnd, prod_bnd, status, wf = param_input.split(well_count, dim=1)

        weights = tc.where(wf < self.stop_well_threshold, self.stop_well_weight, 1.0)
        weights = tc.where(status == 0, self.stop_well_weight, weights)
        
        return tc.cat((weights, weights, weights), 1)
    
    def view(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        well_count = self.well_count
        cell_count = self.cell_count
        state = pmn.ProxyModelNet.State()

        state.pressure, state.x_saturation = input[:, :cell_count * 2].split(cell_count, dim=1)
        state.bhp_bnd, state.status, state.wf = param_input.split(well_count, dim=1)

        self.view_internal(state)

        return tc.cat((state.bhp, state.production - state.production_w, state.production_w), 1)
    
    def update_state(self, state: pmn.ProxyModelNet.State):
        compr = (state.compr_w * state.saturation + state.compr_o * (1.0 - state.saturation))
        dpressure = state.DT * state.Q_in_t / (compr + state.DT * (state.J_t * state.wf + state.S_t))
        pressure_next = state.pressure + dpressure

        state.dpressure_dbhp = state.DT * state.J_t * state.wf / (compr + state.DT * (state.J_t * state.wf + state.S_t))

        state.Q_in_w = state.Q_in_w - (state.J_w * state.wf + state.S_w) * dpressure
        state.dsaturation = (state.DT * state.Q_in_w - state.compr_w * state.saturation * dpressure) / (1.0 - state.DT * state.R_w)

        state.dsaturation_dbph = (state.DT * state.J_w * state.wf - state.compr_w * state.saturation * state.dpressure_dbhp) / (1.0 - state.DT * state.R_w)

        state.pressure = pressure_next    

    def step(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        well_count = self.well_count
        cell_count = self.cell_count
        state = pmn.ProxyModelNet.State()

        state.pressure, state.x_saturation = input[:, :cell_count * 2].split(cell_count, dim=1)
        state.hidden = input[:, self.cell_count * 2:]
        state.status, state.wf = param_input.split(well_count, dim=1)

        state.bhp_bnd = state.pressure[:,:self.well_count]
        bhp_bnd0 = state.bhp_bnd

        self.view_internal0(state)
        self.step_internal(state)

        well_pressure = state.pressure[:,:self.well_count]
        dwellpressure_dbhp = state.dpressure_dbhp[:,:self.well_count]
        well_sat = (state.x_saturation + state.dsaturation)[:,:self.well_count]

        rmob = self.calc_permeability_rel_o(state)
        rmob = tc.where(state.status == 1, rmob, rmob) #tc.full_like(state.status, rmob)
        PI0 = state.J * (1.0 - dwellpressure_dbhp)
        pressure0 = (well_pressure - dwellpressure_dbhp * bhp_bnd0) / (1.0 - dwellpressure_dbhp)
        ds = state.dsaturation_dbph[:,:self.well_count]
        sat0 = well_sat - state.dsaturation_dbph[:,:self.well_count] * bhp_bnd0        

        PI0 = tc.where(state.status == 1, PI0, -PI0)
        sat0 = tc.where(state.status == 1, sat0, 1.0)
        ds = tc.where(state.status == 1, ds, 0.0)
        
        offset_bhp, mult_bhp, offset_prod, mult_prod = self.scales[0], self.scales[1], self.scales[2], self.scales[3]
        
        pressure0_s = pressure0 / mult_bhp - offset_bhp
        PI0_s = PI0 * mult_bhp / mult_prod
        bhp_bnd_s = self.controller.control(PI0_s, pressure0_s, rmob, sat0, ds * mult_bhp)
        state.bhp_bnd = (bhp_bnd_s + offset_bhp) * mult_bhp

        incorrect_status = tc.logical_or((pressure0 - state.bhp_bnd) * state.status < 0.0, state.status == 0)
        state.bhp_bnd = tc.where(incorrect_status, pressure0, state.bhp_bnd)

        dbhp = state.bhp_bnd - bhp_bnd0
        dbhp = self.resize_tensor(dbhp)

        dpressure = state.dpressure_dbhp * dbhp
        state.pressure = state.pressure + dpressure

        dsaturation = state.dsaturation_dbph * dbhp
        state.x_saturation = pmn.ProxyModelNet.mod_sat(state.x_saturation, state.saturation, state.dsaturation + dsaturation)
        
        return tc.cat((state.pressure, state.x_saturation, state.hidden), 1), state.bhp_bnd
