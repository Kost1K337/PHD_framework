import src.inno_reservoir.model.model_base as mb
import torch as tc
import torch.nn as nn
import numpy as np

class ProxyModelNet(mb.IProxyModel):
    stop_well_weight: np.float64
    stop_well_threshold: np.float64
    useNet: bool

    class State:
        cell_count: int
        pressure: tc.Tensor
        x_saturation: tc.Tensor
        hidden: tc.Tensor

        production: tc.Tensor
        bhp: tc.Tensor
        wf: tc.Tensor

        bhp_bnd: tc.Tensor
        prod_bnd: tc.Tensor
        status: tc.Tensor

        production_w: tc.Tensor
        saturation: tc.Tensor
        mob_prod_t: tc.Tensor
        J: tc.Tensor
        J_t: tc.Tensor
        J_w: tc.Tensor
        S_t: tc.Tensor
        S_w: tc.Tensor
        R_t: tc.Tensor
        R_w: tc.Tensor
        Q_in_t: tc.Tensor
        Q_in_w: tc.Tensor

        compr_w: tc.Tensor
        compr_o: tc.Tensor
        DT: tc.Tensor

    def __init__(self, well_count, dt, hidden_count = 0, additionnal_cells = 0):
        super().__init__(well_count, dt)
        self.cell_count = well_count + additionnal_cells
        self.stop_well_weight = 1.0
        self.stop_well_threshold = 1e-6
        self.hidden_count = hidden_count
        self.useNet = True

        cell_count = self.cell_count
        well_state_dim = 2 + hidden_count

        self.flowNet = nn.Sequential(
            nn.Linear(cell_count * well_state_dim, cell_count * well_state_dim * 5, dtype=tc.float32),
            tc.nn.LeakyReLU(),
            nn.Linear(cell_count * well_state_dim * 5, cell_count * well_state_dim * 3, dtype=tc.float32),
            tc.nn.LeakyReLU(),
            nn.Linear(cell_count * well_state_dim * 3, cell_count * well_state_dim, dtype=tc.float32)
        )

        self.register_module("flowNet", self.flowNet)

        for param in self.flowNet.parameters():
            param.data = param / (cell_count * well_state_dim * 100)

        J = tc.rand((well_count), dtype=tc.float32)

        perm = tc.rand((cell_count, cell_count))
        perm = (perm + tc.transpose(perm, 0, 1))

        self.permeability = nn.Parameter(perm)

        self.J = nn.Parameter(J)
        self.compressibility_o = nn.Parameter(tc.tensor(1.0, dtype=tc.float32))
        self.compressibility_w = nn.Parameter(tc.tensor(1.0, dtype=tc.float32))
        self.compressibility_o_grad = nn.Parameter(tc.tensor(0.0001, dtype=tc.float32))
        self.compressibility_w_grad = nn.Parameter(tc.tensor(0.0001, dtype=tc.float32))
        self.volume_inv = nn.Parameter(tc.rand(cell_count, dtype=tc.float32))

        self.permeability_rel_o = nn.Parameter(tc.tensor(1.0, dtype=tc.float32))
        self.permeability_rel_o_g = nn.Parameter(tc.tensor(0.001, dtype=tc.float32))

        self.perm_aqu = nn.Parameter(tc.rand(cell_count, dtype=tc.float32))
        self.bhp_aqu = nn.Parameter(tc.tensor(0.5, dtype=tc.float32))
        self.perm_aqu2 = nn.Parameter(tc.rand(cell_count, dtype=tc.float32))
        self.bhp_aqu2 = nn.Parameter(tc.tensor(0.5, dtype=tc.float32))

        self.pressure_init = nn.Parameter(tc.rand(cell_count, dtype=tc.float32))
        self.saturation_init = nn.Parameter(tc.rand(cell_count, dtype=tc.float32))
        self.hidden_init = nn.Parameter(tc.zeros(cell_count * hidden_count, dtype=tc.float32))

        self.register_parameter("permeability", self.permeability)
        self.register_parameter("J", self.J)
        self.register_parameter("compressibility_o", self.compressibility_o)
        self.register_parameter("compressibility_w", self.compressibility_w)
        self.register_parameter("compressibility_o_grad", self.compressibility_o_grad)
        self.register_parameter("compressibility_w_grad", self.compressibility_w_grad)
        self.register_parameter("volume_inv", self.volume_inv)

        self.register_parameter("permeability_rel_o", self.permeability_rel_o)
        self.register_parameter("permeability_rel_o_g", self.permeability_rel_o_g)

        self.register_parameter("perm_aqu", self.perm_aqu)
        self.register_parameter("bhp_aqu", self.bhp_aqu)
        self.register_parameter("perm_aqu2", self.perm_aqu2)
        self.register_parameter("bhp_aqu2", self.bhp_aqu2)

        self.register_parameter("pressure_init", self.pressure_init)
        self.register_parameter("saturation_init", self.saturation_init)
        self.register_parameter("hidden_init", self.hidden_init)

        self.dt = nn.Parameter(tc.tensor(1.0, dtype=tc.float32), requires_grad=False)
        self.time_steps = nn.Parameter(tc.tensor(1.0, dtype=tc.float32), requires_grad=False)
        self.vol_mult = nn.Parameter(tc.tensor(1.0, dtype=tc.float32), requires_grad=False)
        self.scales = nn.Parameter(tc.zeros(4, dtype=tc.float32), requires_grad=False)

        self.register_parameter("dt", self.dt)
        self.register_parameter("time_steps", self.time_steps)
        self.register_parameter("vol_mult", self.vol_mult)
        self.register_parameter("scales", self.scales)

    def init_params(self, prod_hist: np.ndarray, sat_hist: np.ndarray, bhp_hist: np.ndarray, times: np.ndarray):
        if prod_hist.ndim == 2:
            prod_hist = prod_hist[:, None, :]
        if sat_hist.ndim == 2:
            sat_hist = sat_hist[:, None, :]
        if prod_hist.ndim == 2:
            bhp_hist = bhp_hist[:, None, :]

        time_steps = prod_hist.shape[0]
        well_count = prod_hist.shape[2]   

        self.time_steps.data = tc.tensor(time_steps, dtype=tc.float32)
        self.well_count = well_count  
        cell_count = self.cell_count   

        vol_mult = tc.tensor(1.0 / time_steps, dtype=tc.float32)
        #compr_mult = time_steps        

        self.vol_mult.data = vol_mult
        #self.compr_mult = compr_mult

        dt = (times[-1] - times[0]) / (time_steps - 1)
        self.dt.data = tc.tensor(dt, dtype=tc.float32)

        prod_hist0 = prod_hist.mean(axis=1)
        sat_hist0 = (sat_hist * prod_hist).mean(axis=1) / (prod_hist0 + 1e-12)

        q0 = prod_hist0[:-1]
        q1 = prod_hist0[1:]
        s0 = sat_hist0[:-1]
        s1 = sat_hist0[1:]

        t0 = times[:-1]
        t1 = times[1:]
        dts = t1 - t0

        dob_o = ((q0 * (1.0 - s0) + q1 * (1.0 - s1)) * dts[:,None]).sum() * 0.5

        #St = np.full(well_count, time_steps)

        Sq = prod_hist.mean(axis=(0,1,2))
        Sb = np.nanmean(bhp_hist, axis=(0,1,2))

        #Sb2 = (bhp_hist * bhp_hist).sum(axis=0) / St
        Sqb = np.nanmean(prod_hist * bhp_hist, axis=(0,1,2))
        Sq2 = np.mean(prod_hist * prod_hist, axis=(0,1,2))       

        s0 = sat_hist0[0].mean()
        vol_tot = dob_o / (1.0 - s0 + 0.001)

        p0 = Sb - (Sqb * Sq) / Sq2
        w_count3 = 1.0 - (Sq * Sq) / Sq2
        #w_count3 = w_count3.sum()

        pres0_mean = p0 / w_count3 #p0.sum() / w_count3
        p0 = pres0_mean

        J = Sq2 / (pres0_mean * Sq - Sqb)
        J = np.abs(J)

        #p0 = (Sb * Sq2 - Sqb * Sq) / (Sq2 - Sq * Sq)
        #J = (Sq * Sq - Sq2) / (Sqb - Sb * Sq)
        pres0_mean = p0.mean()

        #J0 = J[J > 0.0].mean()
        #Jstd = J[J > 0.0].std()
        #J = np.where(J < 0.0, J0, J)
        J0 = J
        
        #J = np.where(J > J0 + Jstd * 2.0, J0 + Jstd * 2.0, J)
        #J = np.where(J > J0 + 2 * Jstd, J0, J0)

        Jt = tc.ones(well_count, dtype=tc.float32) * J #tc.from_numpy(J).type(tc.float32)
        self.J.data = tc.sqrt(Jt)

        perm = (tc.rand((cell_count, cell_count)) + 1.0) * J
        perm = (perm + tc.transpose(perm, 0, 1)) * 0.5
        self.permeability.data = tc.sqrt(perm / cell_count)

        #min_compr = tc.abs(Jt) * (self.dt * 2.0)

        #volume = (min_compr / tc.sum(min_compr)) * (vol_tot / vol_mult)
        volume = (vol_tot / cell_count) * tc.ones(cell_count, dtype=tc.float32)
        self.volume_inv.data = 1.0 / tc.sqrt(volume * vol_mult)

        #compressibility = tc.sqrt(min_compr) * 0.1
        compressibility = tc.tensor(0.1, dtype=tc.float32)
        self.compressibility_o.data = compressibility
        self.compressibility_w.data = compressibility * (0.9999)

        self.compressibility_o_grad.data = (compressibility) * 0.0002
        self.compressibility_w_grad.data = (compressibility) * 0.0001

        saturation0 = tc.from_numpy(sat_hist[0]).type(tc.float32)
        saturation0 = tc.zeros(cell_count, dtype=tc.float32)
        #saturation0 = tc.logit((saturation0 + 0.01) / 1.02)
        self.saturation_init.data = saturation0 + 0.1001

        bhp_hist0 = np.nanmean(bhp_hist, axis=1)
        bhp_hist0 = np.where(np.isnan(bhp_hist0), np.nanmean(bhp_hist0, axis=0), bhp_hist0)
        bhp_hist0 = np.where(np.isnan(bhp_hist0), np.nanmean(bhp_hist0), bhp_hist0)

        pressure_well0 = tc.from_numpy(bhp_hist0[0]).type(tc.float32)
        production_well0 = tc.from_numpy(prod_hist0[0]).type(tc.float32)
        pressure_init = pressure_well0 + production_well0 / Jt
        self.pressure_init.data[:well_count] = pressure_init
        self.pressure_init.data[well_count:] = pres0_mean

        perm_aqu = tc.sqrt(tc.rand(cell_count, dtype=tc.float32) + 1.0) * J0
        perm_aqu2 = tc.sqrt(tc.rand(cell_count, dtype=tc.float32) + 1.0) * J0
        bhp_aqu = tc.tensor(pres0_mean).type(tc.float32)

        self.perm_aqu.data = perm_aqu
        self.bhp_aqu.data = bhp_aqu * 0.8
        self.perm_aqu.data = perm_aqu2
        self.bhp_aqu.data = bhp_aqu * 1.2

    def save_scales(self, offset_bhp, mult_bhp, offset_prod, mult_prod):        
        self.scales.data = tc.tensor([offset_bhp, mult_bhp, offset_prod, mult_prod], dtype=tc.float32, device=self.scales.device)

    def get_scales(self):
        scales = self.scales.detach().cpu().numpy()
        offset_bhp, mult_bhp, offset_prod, mult_prod = scales[0], scales[1], scales[2], scales[3]
        return offset_bhp, mult_bhp, offset_prod, mult_prod

    def setNetEnabled(self, useNet):
        self.useNet = useNet
        for param in self.net_params():
            param.requires_grad = useNet
        for param in self.simple_params():
            param.requires_grad = not useNet

    def net_params(self):
        yield  self.hidden_init
        for param in self.flowNet.parameters():
            yield param

    def simple_params(self):
        exclude_params = ["hidden_init", "dt", "time_steps", "vol_mult", "scales"]
        for name, param in self.named_parameters(recurse=False):
            if name in exclude_params:
                continue
            yield param

    def save_base_params(self):
        exclude_params = ["hidden_init", "dt", "time_steps", "vol_mult", "scales"]
        params = {}
        for name, param in self.named_parameters(recurse=False):
            if name in exclude_params:
                continue
            params[name] = param.data.detach().clone()
        return params
    
    def load_base_params(self, params):
        exclude_params = ["hidden_init", "dt", "time_steps", "vol_mult", "scales"]
        for name, param in self.named_parameters(recurse=False):
            if name in exclude_params:
                continue
            param.data = params[name]

    def hidden_dim(self) -> int:
        return self.cell_count * (2 + self.hidden_count)

    def start(self):
        x_sat =  self.saturation_init
        x_sat = tc.relu(x_sat)
        x_sat = 1.0 - tc.relu(1.0 - x_sat)
        return tc.cat((self.pressure_init, x_sat, self.hidden_init), 0)

    def calc_sat(x_saturation: tc.Tensor) -> tc.Tensor:
        saturation = (x_saturation)
        return saturation

    def mod_sat(x_saturation: tc.Tensor, saturation: tc.Tensor, dsaturation: tc.Tensor) -> tc.Tensor:
        x_saturation_next = tc.relu(x_saturation + dsaturation)
        x_saturation_next = 1.0 - tc.relu(1.0 - x_saturation_next)
        return x_saturation_next
    
    def calc_permeabilities_upwind(self, pressure: tc.Tensor) -> tc.Tensor:
        cell_count = self.cell_count

        pressure2 = pressure[:,None,:].expand((-1,  cell_count, cell_count ))
        permeability = self.permeability * self.permeability

        upwind = (pressure2 > tc.transpose(pressure2, 1, 2))
        permeability_out = tc.where(upwind, permeability[None, ...], 0.0)

        return permeability_out
    
    def calc_Q_in_base_perm_m(self, permeability_out: tc.Tensor, pressure: tc.Tensor, mob_prod_t: tc.Tensor):
        pm = pressure * mob_prod_t

        S_out = permeability_out.sum(dim=1)
        R = tc.matmul(pressure[:,None,:], permeability_out)[:,0,:] - S_out * pressure
        S_out = S_out * mob_prod_t
        Q_out = R * mob_prod_t

        S_in = tc.matmul(permeability_out, mob_prod_t[...,None])[...,0]
        Q_in = tc.matmul(permeability_out, pm[...,None])[...,0] - S_in * pressure

        Q = Q_out + Q_in
        S = S_out + S_in
        return Q, S, R
    
    def calc_Q_in_base_perm(self, permeability_out: tc.Tensor, pressure: tc.Tensor, saturation: tc.Tensor, mob_prod_t: tc.Tensor):
        Q_in_t, S_t, R_t = self.calc_Q_in_base_perm_m(permeability_out, pressure, mob_prod_t)
        Q_in_w, S_w, R_w = self.calc_Q_in_base_perm_m(permeability_out, pressure, saturation)
        return Q_in_t, Q_in_w, S_t, S_w, R_t, R_w
    
    def calc_Q_in_base(self, state: State):
        permeability_out = self.calc_permeabilities_upwind(state.pressure)
        Q_in_t0, Q_in_w0, S_t, S_w, R_t, R_w = self.calc_Q_in_base_perm(permeability_out, state.pressure, state.saturation, state.mob_prod_t)
        state.S_t = S_t #- permeability_t_diag
        state.S_w = S_w #- permeability_w_diag
        #state.S_t = 0
        #state.S_w = 0
        state.R_t = R_t
        state.R_w = R_w

        if self.useNet:
            input1 = tc.cat((state.pressure, state.saturation, state.hidden), dim=1)
            out = self.flowNet.forward(input1)
        
            Q_in_o, Q_in_w = out[:, :self.cell_count * 2].split(self.cell_count, dim=1)
            state.hidden = tc.tanh(out[:, self.cell_count * 2:])

            Q_in_o = Q_in_o - Q_in_o.mean(dim=1)[...,None]
            Q_in_w = Q_in_w - Q_in_w.mean(dim=1)[...,None]

            mult_o = tc.relu_(-Q_in_o * (1 - state.saturation)).sum(dim=1) / tc.relu_(-Q_in_o).sum(dim=1)
            mult_w = tc.relu_(-Q_in_w * state.saturation).sum(dim=1) / tc.relu_(-Q_in_w).sum(dim=1)

            Q_in_o = tc.where(Q_in_o < 0, Q_in_o * (1 - state.saturation), Q_in_o * mult_o[...,None])
            Q_in_w = tc.where(Q_in_w < 0, Q_in_w * state.saturation, Q_in_w * mult_w[...,None])

            state.Q_in_t = Q_in_o + Q_in_w + Q_in_t0
            state.Q_in_w = Q_in_w + Q_in_w0
        else:
            state.Q_in_t = Q_in_t0
            state.Q_in_w = Q_in_w0

    def calc_permeability_rel_o(self, state: State):
        permeability_rel_o = self.permeability_rel_o# + self.permeability_rel_o_g * state.pressure
        permeability_rel_o = permeability_rel_o * permeability_rel_o + 1e-1
        return permeability_rel_o
    
    def calc_mobility(self, state: State):
        state.saturation = ProxyModelNet.calc_sat(state.x_saturation)
        permeability_rel_o = self.calc_permeability_rel_o(state)
        state.mob_prod_t = (state.saturation + (1.0 - state.saturation) * permeability_rel_o)
    
    def view_internal(self, state: State):
        self.calc_mobility(state)

        J = self.J * self.J

        state.J_t = tc.where(state.production > 0, J * state.mob_prod_t[:,:self.well_count], J)
        state.J_w = tc.where(state.production > 0, J * state.saturation[:,:self.well_count], J)

        pressure = state.pressure[:,:self.well_count]

        state.bhp = pressure - state.production / state.J_t
        state.production_w = state.J_w * (pressure - state.bhp)

        state.J = tc.zeros_like(J)
        state.J_t = tc.zeros_like(state.J_t)

    def resize_tensor(self, well_tensor: tc.Tensor) -> tc.Tensor:
        if self.cell_count <= self.well_count:
            return well_tensor
        
        pad = tc.zeros((well_tensor.shape[0], self.cell_count - self.well_count), dtype=tc.float32, device=well_tensor.device)
        cell_tensor = tc.cat((well_tensor, pad), 1)
        return cell_tensor
    
    def step_internal(self, state: State):
        self.calc_Q_in_base(state)  

        perm_aqu_abs = self.perm_aqu * self.perm_aqu
        perm_aqu_abs2 = self.perm_aqu2 * self.perm_aqu2

        perm_aqu_abs_t = tc.where(state.pressure > self.bhp_aqu, perm_aqu_abs * state.mob_prod_t, perm_aqu_abs)
        perm_aqu_abs_w = tc.where(state.pressure > self.bhp_aqu, perm_aqu_abs * state.saturation, perm_aqu_abs)
        perm_aqu_abs2_t = tc.where(state.pressure > self.bhp_aqu2, perm_aqu_abs2 * state.mob_prod_t, perm_aqu_abs2)
        perm_aqu_abs2_w = tc.where(state.pressure > self.bhp_aqu2, perm_aqu_abs2 * state.saturation, perm_aqu_abs2)

        prod_aqu_t = perm_aqu_abs_t * (self.bhp_aqu - state.pressure)
        prod_aqu_w = perm_aqu_abs_w * (self.bhp_aqu - state.pressure)
        prod_aqu_t = prod_aqu_t + perm_aqu_abs2_t * (self.bhp_aqu2 - state.pressure)
        prod_aqu_w = prod_aqu_w + perm_aqu_abs2_w * (self.bhp_aqu2 - state.pressure)

        perm_aqu_abs_t = perm_aqu_abs_t + perm_aqu_abs2_t
        perm_aqu_abs_w = perm_aqu_abs_w + perm_aqu_abs2_w

        state.R_w = tc.where(self.bhp_aqu < state.pressure, state.R_w + perm_aqu_abs * (self.bhp_aqu - state.pressure), state.R_w)
        state.R_w = tc.where(self.bhp_aqu2 < state.pressure, state.R_w + perm_aqu_abs2 * (self.bhp_aqu2 - state.pressure), state.R_w)

        prod = self.resize_tensor(state.production * state.wf)
        prod_w = self.resize_tensor(state.production_w * state.wf)

        prod[:,self.well_count:] = 0.0
        prod_w[:,self.well_count:] = 0.0

        state.Q_in_t = state.Q_in_t - prod + prod_aqu_t
        state.Q_in_w = state.Q_in_w - prod_w + prod_aqu_w

        Q0 = state.J * state.wf * (state.pressure[:,:self.well_count] - state.bhp)
        Q0 = self.resize_tensor(Q0)
        state.R_w = tc.where(prod > 0, state.R_w - Q0, state.R_w)

        compr_w =  self.compressibility_w #+ self.compressibility_w_grad * state.pressure
        state.compr_w = compr_w * compr_w

        compr_o =  self.compressibility_o# + self.compressibility_o_grad * state.pressure
        state.compr_o = compr_o * compr_o

        state.J_t = self.resize_tensor(state.J_t)
        state.J_w = self.resize_tensor(state.J_w)
        state.wf = self.resize_tensor(state.wf)

        state.DT = self.dt * self.volume_inv * self.volume_inv * self.vol_mult
        state.S_t = state.S_t + perm_aqu_abs_t
        state.S_w = state.S_w + perm_aqu_abs_w

        self.update_state(state)

    def update_state(self, state: State):
        compr = (state.compr_w * state.saturation + state.compr_o * (1.0 - state.saturation))
        dpressure = state.DT * state.Q_in_t / (compr + state.DT * (state.J_t * state.wf + state.S_t))
        pressure_next = state.pressure + dpressure

        state.Q_in_w = state.Q_in_w - (state.J_w * state.wf + state.S_w) * dpressure

        dsaturation = (state.DT * state.Q_in_w - state.compr_w * state.saturation * dpressure) / (1.0 - state.DT * state.R_w)
        x_saturation_next = ProxyModelNet.mod_sat(state.x_saturation, state.saturation, dsaturation)

        state.pressure, state.x_saturation = pressure_next, x_saturation_next
    
    def veiw_weights(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        well_count = self.well_count
        production, wf = param_input.split(well_count, dim=1)

        weights = tc.where(wf < self.stop_well_threshold, 
                           self.stop_well_weight, 1.0)
        return tc.cat((weights, weights), 1)
    
    def view(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        well_count = self.well_count
        cell_count = self.cell_count
        state = ProxyModelNet.State()
        state.pressure, state.x_saturation = input[:, :cell_count * 2].split(cell_count, dim=1)
        #state.hidden = input[:, self.well_count * 2:]
        state.production, state.wf = param_input.split(well_count, dim=1)
        
        self.view_internal(state)

        return tc.cat((state.bhp, state.production_w), 1)

    def step(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        well_count = self.well_count
        cell_count = self.cell_count
        state = ProxyModelNet.State()

        state.pressure, state.x_saturation = input[:, :cell_count * 2].split(cell_count, dim=1)
        state.hidden = input[:, self.cell_count * 2:]
        state.production, state.wf = param_input.split(well_count, dim=1)   

        self.view_internal(state)
        self.step_internal(state)
        
        return tc.cat((state.pressure, state.x_saturation, state.hidden), 1)