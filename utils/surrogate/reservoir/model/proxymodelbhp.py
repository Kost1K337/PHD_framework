import model.proxymodelnet as pmn
import torch as tc

class ProxyModelNetBHP(pmn.ProxyModelNet):
    def __init__(self, well_count, dt, hidden_count = 0, addition_cells = 0):
        super().__init__(well_count, dt, hidden_count, addition_cells)
        #cell_count

    def view_internal(self, state: pmn.ProxyModelNet.State):
        self.calc_mobility(state)

        pressure = state.pressure[:,:self.well_count]

        J = self.J * self.J
        state.J_t = tc.where(pressure > state.bhp, J * state.mob_prod_t[:,:self.well_count], J)
        state.J_w = tc.where(pressure > state.bhp, J * state.saturation[:,:self.well_count], J)
        state.J = J

        state.production = state.J_t * (pressure - state.bhp)
        state.production_w = state.J_w * (pressure - state.bhp)
    
    def veiw_weights(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        well_count = self.well_count
        bhp, wf, prod_o, prod_w = param_input.split(well_count, dim=1)

        prod = prod_o + prod_w
        
        weights = tc.where(wf < self.stop_well_threshold, self.stop_well_weight, 1.0)        
        weights = tc.where(tc.abs(prod) < self.stop_well_threshold, self.stop_well_weight, weights)
        
        return tc.cat((weights, weights), 1)
    
    def view(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        well_count = self.well_count
        cell_count = self.cell_count
        state = pmn.ProxyModelNet.State()

        state.pressure, state.x_saturation = input[:, :cell_count * 2].split(cell_count, dim=1)
        state.bhp, state.wf = param_input.split(well_count, dim=1)

        self.view_internal(state)

        return tc.cat((state.production - state.production_w, state.production_w), 1)

    def step(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        well_count = self.well_count
        cell_count = self.cell_count
        state = pmn.ProxyModelNet.State()

        state.pressure, state.x_saturation = input[:, :cell_count * 2].split(cell_count, dim=1)
        state.hidden = input[:, self.cell_count * 2:]
        state.bhp, state.wf = param_input.split(well_count, dim=1)

        self.view_internal(state)      
        self.step_internal(state)
        
        return tc.cat((state.pressure, state.x_saturation, state.hidden), 1)
    