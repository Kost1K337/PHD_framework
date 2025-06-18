import model.proxymodelnet as pmn
import torch as tc

class ProxyModelNetMix(pmn.ProxyModelNet):
    def __init__(self, well_count, dt, hidden_count = 0, addition_cells = 0):
        super().__init__(well_count, dt, hidden_count, addition_cells)
        #cell_count

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
        cap = tc.abs(state.production) > tc.abs(state.prod_bnd)
        incorrect_status = tc.logical_or(state.production * state.status < 0.0, state.status == 0)

        state.production = tc.where(cap, state.prod_bnd, state.production)
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
        weights = tc.where(tc.abs(prod_bnd) < self.stop_well_threshold, self.stop_well_weight, weights)
        weights = tc.where(status == 0, self.stop_well_weight, weights)
        
        return tc.cat((weights, weights, weights), 1)
    
    def view(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        well_count = self.well_count
        cell_count = self.cell_count
        state = pmn.ProxyModelNet.State()

        state.pressure, state.x_saturation = input[:, :cell_count * 2].split(cell_count, dim=1)
        state.bhp_bnd, state.prod_bnd, state.status, state.wf = param_input.split(well_count, dim=1)

        self.view_internal(state)

        return tc.cat((state.bhp, state.production - state.production_w, state.production_w), 1)
    
    def update_state(self, state: pmn.ProxyModelNet.State):
        compr = (state.compr_w * state.saturation + state.compr_o * (1.0 - state.saturation))
        dpressure = state.DT * state.Q_in_t / (compr + state.DT * (state.J_t * state.wf + state.S_t))
        pressure_next = state.pressure + dpressure

        state.dpressure_dbhp = state.DT * state.J_t * state.wf / (compr + state.DT * (state.J_t * state.wf + state.S_t))

        state.Q_in_w = state.Q_in_w - (state.J_w * state.wf + state.S_w) * dpressure

        state.dsaturation = (state.DT * state.Q_in_w - state.compr_w * state.saturation * dpressure) / (1.0 - state.DT * state.R_w)
        #x_saturation_next = ProxyModelNet.mod_sat(state.x_saturation, state.saturation, state.dsaturation)

        state.dsaturation_dbph = (state.DT * state.J_w * state.wf - state.compr_w * state.saturation * state.dpressure_dbhp) / (1.0 - state.DT * state.R_w)

        #state.pressure, state.x_saturation = pressure_next, x_saturation_next
        state.pressure = pressure_next

    def step(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        well_count = self.well_count
        cell_count = self.cell_count
        state = pmn.ProxyModelNet.State()

        state.pressure, state.x_saturation = input[:, :cell_count * 2].split(cell_count, dim=1)
        state.hidden = input[:, self.cell_count * 2:]
        state.bhp_bnd, state.prod_bnd, state.status, state.wf = param_input.split(well_count, dim=1)

        self.view_internal0(state)
        self.step_internal(state)

        J_t = state.J_t

        self.view_internal_prod(state)
        cap = tc.abs(state.production) > tc.abs(state.prod_bnd)
        incorrect_status = tc.logical_or(state.production * state.status < 0.0, state.status == 0)

        dprod_dbhp = J_t * (state.dpressure_dbhp - 1.0)
        new_prod = tc.where(cap, state.prod_bnd, state.production)
        new_prod = tc.where(incorrect_status, 0.0, new_prod)

        dbhp = (new_prod - state.production) / dprod_dbhp[:,:self.well_count]
        dbhp = self.resize_tensor(dbhp)

        dpressure = state.dpressure_dbhp * dbhp
        state.pressure = state.pressure + dpressure

        dsaturation = state.dsaturation_dbph * dbhp
        state.x_saturation = pmn.ProxyModelNet.mod_sat(state.x_saturation, state.saturation, state.dsaturation + dsaturation)

        #state.pressure, state.x_saturation = input[:, :cell_count * 2].split(cell_count, dim=1)
        #state.hidden = input[:, self.cell_count * 2:]
        #state.bhp_bnd, state.prod_bnd, state.status, state.wf = param_input.split(well_count, dim=1)

        #self.view_internal_prod(state)
        #pressure = state.pressure[:,:self.well_count]

        #state.production = tc.where(cap, state.prod_bnd, state.production)
        #state.production = tc.where(incorrect_status, 0.0, state.production)
        #state.bhp = pressure - state.production / state.J_t

        #state.production_w = state.J_w * (pressure - state.bhp)

        #zero_grad = tc.logical_or(cap, incorrect_status)
        #state.J_t = tc.where(zero_grad, 0.0, state.J_t)
        #state.J_w = tc.where(zero_grad, 0.0, state.J_w)
        #state.J = tc.where(zero_grad, 0.0, self.J * self.J)

        #self.step_internal(state)
        
        return tc.cat((state.pressure, state.x_saturation, state.hidden), 1)